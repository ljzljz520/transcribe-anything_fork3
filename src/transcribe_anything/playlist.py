"""
Explicit playlist transcription mode.

A playlist is *frozen* before any transcription starts: every entry gets a
fixed position, identity and title, recorded up front in
``playlist_index.json``. Entries are then transcribed strictly sequentially
through the existing single-item backend (``transcribe_anything.api.transcribe``
or the matching remote-client call), so at most one item's download/temp
audio is staged at a time -- the orchestrator never prefetches or buffers
later items.

Per-item output directories are independent and carry the frozen position
prefix (``000_<title>``, ``001_<title>`` ...), so duplicate titles can never
overwrite each other. Entries that are unavailable up front (e.g. a missing
local file) or that raise while being transcribed are recorded in the index
with an explicit status/error and never prevent the remaining entries from
running.
"""

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transcribe_anything.util import sanitize_filename

PLAYLIST_INDEX_NAME = "playlist_index.json"
INDEX_SCHEMA_VERSION = 1

# Item statuses as written into the collection index.
STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = (STATUS_OK, STATUS_UNAVAILABLE, STATUS_FAILED)

# Keep reasons bounded so a chatty backend traceback can't bloat the index.
MAX_ERROR_CHARS = 2000

#: Backend call shape: ``(source, output_dir) -> returned output dir or None``.
TranscribeFn = Callable[[str, str], str | None]

#: A manifest entry may be a bare source string or a dict with explicit
#: identity metadata.
RawEntry = str | dict[str, Any]


@dataclass
class PlaylistEntry:
    """An entry whose identity and order have been frozen before processing."""

    position: int
    item_id: str
    title: str
    source: str


@dataclass
class PlaylistItemResult:
    """Per-item outcome recorded in the collection index."""

    position: int
    item_id: str
    title: str
    source: str
    status: str = STATUS_PENDING
    output_dir: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "id": self.item_id,
            "title": self.title,
            "source": self.source,
            "status": self.status,
            "output_dir": self.output_dir,
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: dict) -> "PlaylistItemResult":
        return PlaylistItemResult(
            position=int(data["position"]),
            item_id=str(data["id"]),
            title=str(data["title"]),
            source=str(data["source"]),
            status=str(data.get("status", STATUS_PENDING)),
            output_dir=data.get("output_dir"),
            error=data.get("error"),
        )


@dataclass
class PlaylistResult:
    """The whole collection: independent per-item results plus index location."""

    collection_dir: str
    items: list[PlaylistItemResult] = field(default_factory=list)
    index_path: str | None = None

    @property
    def ok_count(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_OK)

    @property
    def unavailable_count(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_UNAVAILABLE)

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.items if item.status == STATUS_FAILED)

    @property
    def has_errors(self) -> bool:
        return any(item.status in (STATUS_UNAVAILABLE, STATUS_FAILED) for item in self.items)

    def to_dict(self) -> dict:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "collection": self.collection_dir,
            "item_count": len(self.items),
            "ok_count": self.ok_count,
            "unavailable_count": self.unavailable_count,
            "failed_count": self.failed_count,
            "items": [item.to_dict() for item in self.items],
        }

    @staticmethod
    def from_dict(data: dict, index_path: str | None = None) -> "PlaylistResult":
        return PlaylistResult(
            collection_dir=str(data.get("collection", "")),
            items=[PlaylistItemResult.from_dict(item) for item in data.get("items", [])],
            index_path=index_path,
        )


def _default_title(source: str) -> str:
    """Derive a cheap, network-free display title for a source."""
    if source.startswith(("http://", "https://", "ftp://")):
        tail = source.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
        return tail or "item"
    return os.path.splitext(os.path.basename(source))[0] or "item"


def _default_id(source: str) -> str:
    """Freeze a stable identity from the source itself.

    URLs are identified verbatim; local files are identified by absolute
    path so entries given via different relative paths still share one
    identity. Position always disambiguates repeated entries.
    """
    if source.startswith(("http://", "https://", "ftp://")):
        return source
    return os.path.abspath(source)


def freeze_playlist(entries: Sequence[RawEntry]) -> list[PlaylistEntry]:
    """Fix every entry's position, identity and title up front.

    The returned list owns the order used for both processing and index
    output; later failures never reorder or drop entries. A structurally
    invalid manifest (empty list / entry without a source) raises here,
    before any backend work happens.
    """
    if not isinstance(entries, list) or not entries:
        raise ValueError("Playlist manifest must contain at least one entry.")
    frozen: list[PlaylistEntry] = []
    for position, raw in enumerate(entries):
        item_id: str | None
        title: str | None
        if isinstance(raw, str):
            source = raw.strip()
            title = None
            item_id = None
        elif isinstance(raw, dict):
            source = str(raw.get("source") or raw.get("url") or raw.get("file") or "").strip()
            raw_title = raw.get("title")
            title = str(raw_title).strip() if raw_title is not None and str(raw_title).strip() else None
            raw_id = raw.get("id") or raw.get("item_id")
            item_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None
        else:
            raise TypeError(f"Playlist entry #{position} must be a string or object, got {type(raw).__name__}.")
        if not source:
            raise ValueError(f"Playlist entry #{position} is missing a source/url/file.")
        frozen.append(
            PlaylistEntry(
                position=position,
                item_id=item_id or _default_id(source),
                title=title or _default_title(source),
                source=source,
            )
        )
    return frozen


def load_playlist_manifest(path: str | Path) -> list[RawEntry]:
    """Load a raw playlist manifest from disk.

    Supported formats:

    * ``.json`` -- a JSON list of source strings and/or
      ``{"source": ..., "title": ..., "id": ...}`` objects; a wrapper object
      with an ``items``/``playlist``/``entries`` list is also accepted.
    * anything else (``.txt``/``.m3u``/``.m3u8``) -- one source per line;
      blank lines and ``#`` comments are ignored. A trailing tab-separated
      value on a line is taken as the explicit title.
    """
    manifest = Path(path)
    text = manifest.read_text(encoding="utf-8")
    if manifest.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("items", "playlist", "entries"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                raise ValueError("JSON playlist manifest must be a list or contain an 'items' list.")
        if not isinstance(data, list):
            raise ValueError("JSON playlist manifest must be a list of entries.")
        return data

    entries: list[RawEntry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            source, title = line.split("\t", 1)
            entries.append({"source": source.strip(), "title": title.strip()})
        else:
            entries.append(line)
    return entries


def _unique_dirname(entry: PlaylistEntry, used: set[str], width: int) -> str:
    """Build the per-item output directory name.

    The frozen position prefix is what makes duplicate titles coexist on
    disk (``000_Keynote`` vs ``001_Keynote``); the used-name set is a
    belt-and-braces guard against sanitization collisions.
    """
    base = sanitize_filename(entry.title) or "item"
    prefix = f"{entry.position:0{width}d}_"
    stem = prefix + base
    candidate = stem
    dedupe = 2
    while candidate in used:
        candidate = f"{stem}_{dedupe}"
        dedupe += 1
    used.add(candidate)
    return candidate


def _preflight_unavailable(source: str) -> str | None:
    """Return a reason string when an entry can't be handed to the backend.

    Local sources must exist as files. URL availability is left to the
    backend (a network probe would stage nothing but would also be slow
    and unreliable); failures there land in STATUS_FAILED instead.
    """
    if source.startswith(("http://", "https://", "ftp://")):
        return None
    if not os.path.isfile(source):
        return f"local file not found: {source}"
    return None


def _write_index_atomic(index_path: Path, result: PlaylistResult) -> None:
    """Rewrite the collection index atomically so an interrupted run leaves
    a valid index describing everything completed so far."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=index_path.name + ".", suffix=".tmp", dir=str(index_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
        os.replace(tmp_name, index_path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def read_playlist_index(path: str | Path) -> PlaylistResult:
    """Load a previously written collection index."""
    index_path = Path(path)
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return PlaylistResult.from_dict(data, index_path=str(index_path.resolve()))


def transcribe_playlist(
    entries: Sequence[RawEntry],
    collection_dir: str | Path,
    *,
    transcribe_fn: TranscribeFn,
) -> PlaylistResult:
    """Transcribe a frozen playlist strictly sequentially.

    Processing is one item at a time and the backend call for an item is
    the only thing that may stage that item's audio, so peak transient
    storage is bounded to a single entry regardless of playlist length.

    Args:
        entries: Raw or already-frozen manifest entries.
        collection_dir: Directory that holds the per-item output dirs and
            ``playlist_index.json``.
        transcribe_fn: Existing single-item backend, called as
            ``transcribe_fn(source, item_output_dir)``. May return the
            resolved output dir or ``None``.

    Returns:
        PlaylistResult with one entry per frozen item, in frozen order.
    """
    frozen = freeze_playlist(entries)
    collection = Path(collection_dir)
    collection.mkdir(parents=True, exist_ok=True)
    index_path = collection / PLAYLIST_INDEX_NAME
    width = max(3, len(str(len(frozen))))

    results = [
        PlaylistItemResult(
            position=entry.position,
            item_id=entry.item_id,
            title=entry.title,
            source=entry.source,
        )
        for entry in frozen
    ]
    result = PlaylistResult(
        collection_dir=str(collection.resolve()),
        items=results,
        index_path=str(index_path.resolve()),
    )
    # Freeze identities/order on disk before touching the first backend.
    _write_index_atomic(index_path, result)
    print(f"[playlist] frozen {len(frozen)} entries in {collection}")

    used_names: set[str] = set()
    for entry, item in zip(frozen, results):
        dirname = _unique_dirname(entry, used_names, width)
        item_dir = str(collection / dirname)
        item.output_dir = str(Path(item_dir).resolve())
        label = f"[playlist] ({entry.position + 1}/{len(frozen)})"
        print(f"{label} {entry.title!r} -> {item_dir}")

        reason = _preflight_unavailable(entry.source)
        if reason is not None:
            item.status = STATUS_UNAVAILABLE
            item.error = reason
            print(f"{label} unavailable: {reason}")
        else:
            try:
                returned = transcribe_fn(entry.source, item_dir)
                if returned:
                    item.output_dir = str(Path(returned).resolve())
                item.status = STATUS_OK
                print(f"{label} done -> {item.output_dir}")
            except KeyboardInterrupt:
                item.status = STATUS_FAILED
                item.error = "aborted by user (KeyboardInterrupt)"
                _write_index_atomic(index_path, result)
                raise
            except Exception as exc:  # noqa: BLE001 - recorded per item, run continues
                message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
                item.status = STATUS_FAILED
                item.error = message
                print(f"{label} failed: {message}")
        # Rewrite after every item: the index always reflects true progress.
        _write_index_atomic(index_path, result)

    print(f"[playlist] finished: {result.ok_count} ok, " f"{result.unavailable_count} unavailable, {result.failed_count} failed")
    return result


def run_playlist_manifest(
    manifest_path: str | Path,
    collection_dir: str | Path,
    *,
    transcribe_fn: TranscribeFn,
) -> PlaylistResult:
    """Load a manifest file, freeze it and run it sequentially."""
    raw_entries = load_playlist_manifest(manifest_path)
    return transcribe_playlist(raw_entries, collection_dir, transcribe_fn=transcribe_fn)
