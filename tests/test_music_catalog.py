import base64
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.providers.media_ffmpeg import generate_tone
from backend.app.providers.music import select_music
from backend.app.services.music_library import import_music, list_music_library


class MusicCatalogTests(unittest.TestCase):
    def test_imported_audio_is_reusable_and_retry_does_not_duplicate_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            generate_tone(source, 0.2, 220, 0.1)
            catalog = root / "catalog.json"
            music = root / "music"
            data = dict(name="湖畔旁白配乐.wav", mood="雅致", style="国风雅韵",
                        license="原创", author="本地作者", usage_scope="比赛展示", approved=True,
                        data_base64=base64.b64encode(source.read_bytes()).decode("ascii"))

            with self.assertRaisesRegex(ValueError, "授权"):
                import_music(dict(data, approved=False), catalog, music)
            with self.assertRaisesRegex(ValueError, "解码"):
                import_music(dict(data, data_base64=base64.b64encode(b"x" * 1024).decode()), catalog, music)
            self.assertFalse(catalog.exists())

            track = import_music(data, catalog, music)
            self.assertEqual(import_music(data, catalog, music), track)
            self.assertEqual(list_music_library(catalog, music)["items"], [track])
            self.assertEqual(len(list((music / "imported").iterdir())), 1)
            selected = select_music(
                task_audio={}, mood="雅致", style="国风雅韵", duration=15, catalog_path=catalog, music_dir=music,
            )
            self.assertEqual(selected.id, track["id"])
            self.assertEqual(selected.title, "湖畔旁白配乐")
            self.assertTrue(Path(selected.path).is_file())

    def test_only_approved_exact_mood_and_style_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "licensed.wav"
            generate_tone(track, 1, 220, 0.1)
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "tracks": [
                            {
                                "id": "licensed",
                                "file": track.name,
                                "moods": ["舒缓"],
                                "styles": ["诗意纪实"],
                                "license": "team-owned",
                                "author": "team",
                                "usage_scope": "competition",
                                "approved": True,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            selected = select_music(
                task_audio={}, mood="舒缓", style="诗意纪实", duration=2,
                catalog_path=catalog, music_dir=root,
            )
            self.assertEqual(selected.id, "licensed")
            with self.assertRaisesRegex(RuntimeError, "No approved music"):
                select_music(
                    task_audio={}, mood="热烈", style="诗意纪实", duration=2,
                    catalog_path=catalog, music_dir=root,
                )


if __name__ == "__main__":
    unittest.main()


class PinnedMusicTrackTests(unittest.TestCase):
    """用户点了具体曲目：必须放这一首，而不是同情绪里的随机一首。"""

    def _catalog(self, root: Path) -> Path:
        (root / "calm-a.wav").write_bytes(b"RIFFtiny")
        (root / "calm-b.wav").write_bytes(b"RIFFtiny")
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps({
            "tracks": [
                {"id": "calm-a", "file": "calm-a.wav", "moods": ["舒缓"],
                 "styles": ["诗意纪实"], "license": "原创", "author": "作者A",
                 "usage_scope": "比赛展示", "approved": True},
                {"id": "calm-b", "file": "calm-b.wav", "moods": ["舒缓"],
                 "styles": ["诗意纪实"], "license": "原创", "author": "作者B",
                 "usage_scope": "比赛展示", "approved": True},
            ]
        }, ensure_ascii=False), encoding="utf-8")
        return catalog

    def test_pinned_track_is_exactly_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            for _ in range(6):
                result = select_music(
                    task_audio={}, mood="舒缓", style="诗意纪实", duration=10,
                    catalog_path=catalog, music_dir=root, track_id="calm-b",
                )
                self.assertEqual(result.id, "calm-b")
                self.assertTrue(result.event.get("pinned"))

    def test_pinned_track_ignores_mood_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            result = select_music(
                task_audio={}, mood="热烈", style="诗意纪实", duration=10,
                catalog_path=catalog, music_dir=root, track_id="calm-a",
            )
            self.assertEqual(result.id, "calm-a", "锁定曲目不受情绪筛选限制")

    def test_unknown_pinned_track_falls_back_with_note(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            result = select_music(
                task_audio={}, mood="舒缓", style="诗意纪实", duration=10,
                catalog_path=catalog, music_dir=root, track_id="missing-id",
            )
            self.assertIn(result.id, {"calm-a", "calm-b"})
            self.assertIn("pinned_missing", result.event)


class TaskInputMusicTrackTests(unittest.TestCase):
    def test_music_track_id_roundtrip(self):
        from backend.app.models import TaskInput

        value = TaskInput.from_dict(
            {"city": "杭州", "theme": "主题", "music_track_id": "  bgm-x  "}
        )
        self.assertEqual(value.music_track_id, "bgm-x")
        self.assertEqual(TaskInput.from_dict({"city": "杭州", "theme": "主题"}).music_track_id, "")

    def test_music_track_id_length_guard(self):
        from backend.app.models import TaskInput

        with self.assertRaises(ValueError):
            TaskInput.from_dict({"city": "杭州", "theme": "主题", "music_track_id": "x" * 201})
