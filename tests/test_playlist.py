"""
Tests for the explicit playlist transcription mode.

The real transcription backend is replaced by a fake that writes an
independent ``out.srt`` per call, so reordering, duplicate titles and
partial failures can be exercised without any network/model access.
"""

# pylint: disable=bad-option-value,useless-option-value,no-self-use,protected-access,R0801
# flake8: noqa E501

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcribe_anything import playlist
from transcribe_anything.playlist import (
    PLAYLIST_INDEX_NAME,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_UNAVAILABLE,
    freeze_playlist,
    load_playlist_manifest,
    read_playlist_index,
    run_playlist_manifest,
    transcribe_playlist,
)


class FakeBackend:
    """Mock single-item backend: writes a distinct out.srt per call.

    Mirrors ``api.transcribe``'s contract ``(source, output_dir) -> dir``.

    fail_sources: sources for which the backend raises (partial failure).
    collection + check_lazy_staging: assert that no later-position item
        directory has been materialized yet at call time (nothing is
        prefetched or staged up front).
    freeze_check: callback invoked on the first call, once the
        pre-transcription pending index exists.
    """

    def __init__(self, *, fail_sources=None, collection=None, check_lazy_staging=False, freeze_check=None):
        self.fail_sources = set(fail_sources or ())
        self.collection = collection
        self.check_lazy_staging = check_lazy_staging
        self.freeze_check = freeze_check
        self.calls = []
        self._freeze_checked = False

    def __call__(self, source, output_dir):
        if self.check_lazy_staging and self.collection is not None:
            current_pos = int(Path(output_dir).name.split("_", 1)[0])
            existing = [name for name in os.listdir(self.collection) if name[:3].isdigit()]
            later = [name for name in existing if int(name.split("_", 1)[0]) >= current_pos]
            assert not later, f"item {current_pos} staged before its turn: {later}"
        if self.freeze_check is not None and not self._freeze_checked:
            self._freeze_checked = True
            self.freeze_check()
        self.calls.append((source, output_dir))
        if source in self.fail_sources:
            raise RuntimeError(f"simulated backend failure for {source}")
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "out.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n" + source + "\n",
            encoding="utf-8",
        )
        return output_dir


class FreezePlaylistTester(unittest.TestCase):
    """Identity/order freezing and manifest parsing."""

    def test_positions_identities_and_default_titles_frozen(self) -> None:
        sources = [
            "https://example.com/a",
            "https://example.com/b?x=1",
            str(Path("nonexistent-dir") / "clip.mp3"),
        ]
        entries = freeze_playlist(sources)
        self.assertEqual([e.position for e in entries], [0, 1, 2])
        self.assertEqual([e.source for e in entries], sources)
        self.assertEqual(entries[0].item_id, sources[0])
        self.assertEqual(entries[1].item_id, sources[1])
        self.assertEqual(entries[2].item_id, os.path.abspath(sources[2]))
        self.assertEqual([e.title for e in entries], ["a", "b", "clip"])

    def test_reordering_remaps_positions(self) -> None:
        sources = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
        forward = freeze_playlist(sources)
        reordered = freeze_playlist([sources[2], sources[0], sources[1]])
        self.assertEqual([e.source for e in reordered], [sources[2], sources[0], sources[1]])
        self.assertEqual([e.position for e in reordered], [0, 1, 2])
        # The same source carries whichever frozen position the manifest gave it.
        self.assertEqual(reordered[1].item_id, forward[0].item_id)
        self.assertEqual(reordered[1].position, 1)
        self.assertEqual(forward[0].position, 0)

    def test_explicit_id_and_title(self) -> None:
        entries = freeze_playlist([{"id": "ep-12", "title": "Show / Intro:", "source": "https://example.com/x"}])
        self.assertEqual(entries[0].item_id, "ep-12")
        self.assertEqual(entries[0].title, "Show / Intro:")

    def test_invalid_manifests_raise_before_processing(self) -> None:
        with self.assertRaises(ValueError):
            freeze_playlist([])
        with self.assertRaises(ValueError):
            freeze_playlist([{"title": "no source"}])
        with self.assertRaises((TypeError, ValueError)):
            freeze_playlist([123])  # type: ignore[list-item]

    def test_load_manifest_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_strings = Path(tmp, "s.json")
            json_strings.write_text(json.dumps(["https://example.com/a", "https://example.com/b"]), encoding="utf-8")
            self.assertEqual(len(load_playlist_manifest(json_strings)), 2)

            json_objects = Path(tmp, "o.json")
            json_objects.write_text(
                json.dumps({"items": [{"source": "https://example.com/a", "title": "A"}]}),
                encoding="utf-8",
            )
            loaded = load_playlist_manifest(json_objects)
            self.assertEqual(loaded[0], {"source": "https://example.com/a", "title": "A"})

            txt = Path(tmp, "list.txt")
            txt.write_text(
                "# comment line\n\nhttps://example.com/a\n" "https://example.com/b\tTitled Entry\n",
                encoding="utf-8",
            )
            loaded_txt = load_playlist_manifest(txt)
            self.assertEqual(loaded_txt[0], "https://example.com/a")
            self.assertEqual(loaded_txt[1], {"source": "https://example.com/b", "title": "Titled Entry"})

            m3u = Path(tmp, "list.m3u")
            m3u.write_text("#EXTM3U\nhttps://example.com/a\n", encoding="utf-8")
            self.assertEqual(load_playlist_manifest(m3u), ["https://example.com/a"])

            bad = Path(tmp, "bad.json")
            bad.write_text(json.dumps({"not_items": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_playlist_manifest(bad)


class TranscribePlaylistTester(unittest.TestCase):
    """End-to-end orchestration with a mock backend."""

    def test_duplicate_titles_get_independent_results(self) -> None:
        entries: list[str | dict[str, str]] = [
            {"source": "https://example.com/first", "title": "Keynote"},
            {"source": "https://example.com/second", "title": "Keynote"},
            "https://example.com/first",  # same source repeated, default title
        ]
        with tempfile.TemporaryDirectory() as collection:
            backend = FakeBackend()
            result = transcribe_playlist(entries, collection, transcribe_fn=backend)

            self.assertEqual(result.ok_count, 3)
            statuses = [item.status for item in result.items]
            self.assertEqual(statuses, [STATUS_OK, STATUS_OK, STATUS_OK])
            dirs = [Path(item.output_dir) for item in result.items if item.output_dir is not None]
            self.assertEqual(len(dirs), 3)
            self.assertEqual([d.name for d in dirs], ["000_Keynote", "001_Keynote", "002_first"])
            for directory in dirs:
                self.assertTrue(directory.is_dir())
                self.assertTrue((directory / "out.srt").is_file())
            # Independent results: identical titles keep distinct contents.
            self.assertNotEqual(
                (dirs[0] / "out.srt").read_text(encoding="utf-8"),
                (dirs[1] / "out.srt").read_text(encoding="utf-8"),
            )
            # Backend was invoked in frozen order with distinct targets.
            self.assertEqual([source for source, _ in backend.calls], [e["source"] if isinstance(e, dict) else e for e in entries])
            self.assertEqual(len({out_dir for _, out_dir in backend.calls}), 3)

    def test_identities_frozen_on_disk_before_first_backend_call(self) -> None:
        entries = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]
        with tempfile.TemporaryDirectory() as collection:

            def assert_pending_index():
                index = read_playlist_index(Path(collection, PLAYLIST_INDEX_NAME))
                self.assertEqual([item.source for item in index.items], entries)
                self.assertTrue(all(item.status == STATUS_PENDING for item in index.items))
                self.assertEqual([item.position for item in index.items], [0, 1, 2])

            backend = FakeBackend(freeze_check=assert_pending_index)
            result = transcribe_playlist(entries, collection, transcribe_fn=backend)
            self.assertEqual(result.ok_count, 3)

    def test_partial_failure_and_unavailable_do_not_stop_the_rest(self) -> None:
        missing_local = str(Path(tempfile.gettempdir()) / "definitely-missing-clip-9f3c.mp3")
        entries = [
            {"source": "https://example.com/good1", "title": "Good One"},
            {"source": "https://example.com/broken", "title": "Broken"},
            {"source": missing_local, "title": "Missing Local"},
            {"source": "https://example.com/good2", "title": "Good Two"},
        ]
        with tempfile.TemporaryDirectory() as collection:
            backend = FakeBackend(
                fail_sources={"https://example.com/broken"},
                collection=collection,
                check_lazy_staging=True,
            )
            result = transcribe_playlist(entries, collection, transcribe_fn=backend)

            by_status = [(item.title, item.status) for item in result.items]
            self.assertEqual(
                by_status,
                [
                    ("Good One", STATUS_OK),
                    ("Broken", STATUS_FAILED),
                    ("Missing Local", STATUS_UNAVAILABLE),
                    ("Good Two", STATUS_OK),
                ],
            )
            self.assertTrue(result.has_errors)
            self.assertEqual(result.ok_count, 2)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(result.unavailable_count, 1)

            failed = result.items[1]
            failed_error = failed.error
            assert failed_error is not None
            self.assertIn("simulated backend failure", failed_error)
            unavailable = result.items[2]
            unavailable_error = unavailable.error
            assert unavailable_error is not None
            self.assertIn("local file not found", unavailable_error)
            called_sources = [source for source, _ in backend.calls]
            self.assertNotIn(missing_local, called_sources)  # never handed to backend
            self.assertEqual(
                called_sources,
                ["https://example.com/good1", "https://example.com/broken", "https://example.com/good2"],
            )
            # Sequential calls; good2 ran despite the preceding failures.
            good1_dir = result.items[0].output_dir
            good2_dir = result.items[3].output_dir
            assert good1_dir is not None
            assert good2_dir is not None
            self.assertTrue(Path(good1_dir, "out.srt").is_file())
            self.assertTrue(Path(good2_dir, "out.srt").is_file())
            # Unavailable items stage nothing; failed mock raised before creating its dir.
            self.assertFalse(Path(unavailable.output_dir or "").exists())
            self.assertFalse(Path(failed.output_dir or "").exists())

    def test_index_persisted_and_roundtrips_in_frozen_order(self) -> None:
        entries = [
            {"source": "https://example.com/c", "title": "C"},
            {"source": "https://example.com/a", "title": "A"},
        ]
        with tempfile.TemporaryDirectory() as collection:
            transcribe_playlist(entries, collection, transcribe_fn=FakeBackend())
            index_path = Path(collection, PLAYLIST_INDEX_NAME)
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], playlist.INDEX_SCHEMA_VERSION)
            self.assertEqual(raw["item_count"], 2)
            self.assertEqual(raw["ok_count"], 2)
            self.assertEqual([item["title"] for item in raw["items"]], ["C", "A"])
            reloaded = read_playlist_index(index_path)
            self.assertEqual(reloaded.collection_dir, str(Path(collection).resolve()))
            self.assertEqual([item.position for item in reloaded.items], [0, 1])
            self.assertTrue(all(item.status == STATUS_OK for item in reloaded.items))

    def test_staging_is_lazy_and_bounded_to_current_item(self) -> None:
        missing_local = str(Path(tempfile.gettempdir()) / "definitely-missing-clip-7a21.mp3")
        entries = [
            "https://example.com/a",
            missing_local,  # unavailable in the middle
            "https://example.com/b",
            "https://example.com/c",
        ]
        with tempfile.TemporaryDirectory() as collection:
            backend = FakeBackend(collection=collection, check_lazy_staging=True)
            result = transcribe_playlist(entries, collection, transcribe_fn=backend)
            # Only the three processed items materialized dirs; nothing prefetched.
            staged = [name for name in os.listdir(collection) if name[:3].isdigit()]
            self.assertEqual(len(staged), 3)
            self.assertEqual(result.ok_count, 3)
            self.assertEqual(result.unavailable_count, 1)

    def test_run_manifest_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp, "list.json")
            manifest.write_text(json.dumps(["https://example.com/a", "https://example.com/b"]), encoding="utf-8")
            collection = Path(tmp, "out")
            result = run_playlist_manifest(manifest, collection, transcribe_fn=FakeBackend())
            self.assertEqual(result.ok_count, 2)
            self.assertTrue((collection / PLAYLIST_INDEX_NAME).is_file())


class PlaylistCliCompatibilityTester(unittest.TestCase):
    """Single-file mode must remain unchanged when --playlist is absent."""

    def test_parse_defaults_keep_single_file_mode(self) -> None:
        self.assertTrue(hasattr(playlist, "transcribe_playlist"))
        from transcribe_anything._cmd import parse_arguments

        with patch.object(sys, "argv", ["transcribe_anything", "video.mp4"]):
            parsed = parse_arguments()
        self.assertIsNone(parsed.playlist)
        self.assertEqual(parsed.url_or_file, "video.mp4")

        with patch.object(sys, "argv", ["transcribe_anything", "--playlist", "list.json"]):
            parsed = parse_arguments()
        self.assertEqual(parsed.playlist, "list.json")
        self.assertIsNone(parsed.url_or_file)

    def test_main_rejects_playlist_combined_with_positional(self) -> None:
        from transcribe_anything import _cmd

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp, "list.json")
            manifest.write_text(json.dumps(["https://example.com/a"]), encoding="utf-8")
            with patch.object(sys, "argv", ["transcribe_anything", "video.mp4", "--playlist", str(manifest)]):
                with patch.object(_cmd, "detect_macos_x86_unsupported", return_value=None):
                    rc = _cmd.main()
            self.assertEqual(rc, 1)
            # The positional must not have triggered any collection output.
            self.assertFalse((Path.cwd() / "text_playlist_list").exists())

    def test_main_runs_playlist_through_existing_backend(self) -> None:
        from transcribe_anything import _cmd

        def fake_transcribe(url_or_file, output_dir, **_kwargs):
            os.makedirs(output_dir, exist_ok=True)
            Path(output_dir, "out.srt").write_text(url_or_file, encoding="utf-8")
            return output_dir

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp, "list.json")
            collection = Path(tmp, "collection")
            manifest.write_text(
                json.dumps(["https://example.com/a", "https://example.com/b"]),
                encoding="utf-8",
            )
            argv = [
                "transcribe_anything",
                "--playlist",
                str(manifest),
                "--output_dir",
                str(collection),
                "--device",
                "cpu",
                "--model",
                "tiny",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(_cmd, "detect_macos_x86_unsupported", return_value=None):
                    with patch("transcribe_anything.api.transcribe", fake_transcribe):
                        rc = _cmd.main()
            self.assertEqual(rc, 0)
            index = read_playlist_index(collection / PLAYLIST_INDEX_NAME)
            self.assertEqual(index.ok_count, 2)
            self.assertTrue((collection / "000_a" / "out.srt").is_file())
            self.assertTrue((collection / "001_b" / "out.srt").is_file())


if __name__ == "__main__":
    unittest.main()
