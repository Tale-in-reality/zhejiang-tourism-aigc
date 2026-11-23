"""视频分辨率协商与负向词折入的单元测试。

背景：网关上的 minimax-h3 只接受 768P，而用户选的成片规格仍是 480P/720P/1080P。
协商层负责"请求档 → 模型支持档"的映射并如实记录，绝不静默失败。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.providers.video import (
    AssetResult,
    CloudVideoProvider,
    MODEL_RESOLUTIONS,
    negotiate_resolution,
    normalize_resolution,
)


class NegotiateResolutionTests(unittest.TestCase):
    def test_supported_value_passes_through(self):
        self.assertEqual(negotiate_resolution("768p", "minimax-h3")[:2], ("768p", False))

    def test_720p_maps_to_768p(self):
        value, mapped, note = negotiate_resolution("720p", "minimax-h3")
        self.assertEqual(value, "768p")
        self.assertTrue(mapped)
        self.assertIn("768P", note)

    def test_1080p_maps_to_768p(self):
        value, mapped, _ = negotiate_resolution("1080p", "minimax-h3")
        self.assertEqual(value, "768p")
        self.assertTrue(mapped)

    def test_480p_maps_to_768p(self):
        value, mapped, _ = negotiate_resolution("480p", "minimax-h3")
        self.assertEqual(value, "768p")
        self.assertTrue(mapped)

    def test_unknown_model_is_untouched(self):
        self.assertEqual(negotiate_resolution("720p", "another-model")[:2], ("720p", False))

    def test_prefers_smallest_supported_at_or_above_request(self):
        with patch.dict(MODEL_RESOLUTIONS, {"test-model": ("480p", "720p", "1080p")}, clear=False):
            self.assertEqual(negotiate_resolution("720p", "test-model")[:2], ("720p", False))
            self.assertEqual(negotiate_resolution("768p", "test-model")[:2], ("1080p", True))

    def test_supported_set_stays_valid(self):
        for model, values in MODEL_RESOLUTIONS.items():
            for value in values:
                self.assertEqual(normalize_resolution(value), value, f"{model}: {value}")


class _FakeTokendance(CloudVideoProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://tokendance.space/gateway/minimax/v2",
            api_key="sk-test",
            model="minimax-h3",
            **kwargs,
        )
        self.captured = None

    def _generate_tokendance(self, shot, frame_path, output_dir, prompt, seed, duration, resolution, ratio=""):
        self.captured = {
            "prompt": prompt,
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
        }
        return AssetResult(
            path=str(Path(output_dir) / "shot.mp4"),
            media_type="video",
            source=self.name,
            model=self.model,
            seed=seed,
            prompt=prompt,
            event={"type": "video", "provider": self.name},
        )


class VideoGenerateWiringTests(unittest.TestCase):
    def test_720p_request_is_negotiated_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _FakeTokendance(require_first_frame=False)
            result = provider.generate(
                {
                    "id": "shot_1",
                    "order": 1,
                    "task_id": "task_x",
                    "visual_prompt": "断桥晨雾，中景平视",
                    "duration_sec": 5,
                },
                Path(directory),
                {
                    "resolution": "720p",
                    "ratio": "9:16",
                    "start_reference": "",
                },
            )
            self.assertEqual(provider.captured["resolution"], "768p", "应协商到模型支持档")
            self.assertGreaterEqual(provider.captured["duration"], 5)
            self.assertEqual(result.event["resolution_requested"], "720p")
            self.assertEqual(result.event["resolution_used"], "768p")
            self.assertTrue(result.event["resolution_mapped"])
            self.assertIn("768P", result.event["resolution_note"])

    def test_negative_prompt_folded_into_tokendance_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _FakeTokendance(require_first_frame=False)
            provider.generate(
                {
                    "id": "shot_1",
                    "order": 1,
                    "task_id": "task_x",
                    "visual_prompt": "断桥晨雾，中景平视",
                    "duration_sec": 5,
                },
                Path(directory),
                {"resolution": "1080p", "ratio": "9:16", "negative_prompt": "文字，水印，畸形手指"},
            )
            self.assertIn("不要出现以下内容：文字，水印，畸形手指", provider.captured["prompt"])
            self.assertEqual(provider.captured["resolution"], "768p")


if __name__ == "__main__":
    unittest.main()
