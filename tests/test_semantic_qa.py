"""语义验收（VLM）与自动重生成回路的单元测试。

分三层：
1. 判定归一化（normalize_verdict）——通过/不通过的每条口径；
2. 判定指令与返工要求（build_shot_instruction / regeneration_clause）；
3. 接线：未配置视觉模型时 orchestrator 必须报 unconfigured（不得冒充通过），
   配置后用桩 VLM 跑通"判定 → 记阻断 → 尝试重生成"。
"""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from backend.app.config import Settings
from backend.app.models import AssetRecord, Shot, Task, TaskInput, TraceRecord
from backend.app.providers.stock import StockProvider
from backend.app.providers.vision import (
    VisionProvider,
    build_vision_message,
    build_vision_provider,
    parse_json_object,
)
from backend.app.services import semantic_qa
from backend.app.services.orchestrator import Orchestrator
from backend.app.storage import Storage


class NormalizeVerdictTests(unittest.TestCase):
    def test_clean_verdict_passes(self):
        verdict = semantic_qa.normalize_verdict(
            {
                "matches_shot": True,
                "subject_consistent": 5,
                "violations": [],
                "location_plausible": True,
                "note": "断桥石栏与晨雾一致",
            }
        )
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["reasons"], [])
        self.assertEqual(verdict["subject_consistent"], 5)

    def test_mismatch_fails(self):
        verdict = semantic_qa.normalize_verdict({"matches_shot": False, "subject_consistent": 5})
        self.assertFalse(verdict["passed"])
        self.assertIn("画面与分镜描述不符", verdict["reasons"])

    def test_blocking_violation_fails_and_is_named(self):
        verdict = semantic_qa.normalize_verdict(
            {"matches_shot": True, "subject_consistent": 4,
             "violations": ["modern_intrusion", "garbled_text"]}
        )
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["blocking_violations"], ["modern_intrusion", "garbled_text"])
        self.assertTrue(any("现代物闯入" in reason for reason in verdict["reasons"]))

    def test_warning_only_violation_passes(self):
        verdict = semantic_qa.normalize_verdict(
            {"matches_shot": True, "subject_consistent": 4, "violations": ["extra_people", "low_quality"]}
        )
        self.assertTrue(verdict["passed"], "提醒级违规不应阻断")
        self.assertEqual(sorted(verdict["warnings"]), ["extra_people", "low_quality"])

    def test_low_subject_score_fails_with_threshold(self):
        verdict = semantic_qa.normalize_verdict(
            {"matches_shot": True, "subject_consistent": 2}, min_subject_score=3
        )
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("主体一致性" in reason for reason in verdict["reasons"]))

    def test_location_rejected_fails_only_when_checked(self):
        raw = {"matches_shot": True, "subject_consistent": 5, "location_plausible": False}
        self.assertFalse(semantic_qa.normalize_verdict(raw, check_location=True)["passed"])
        self.assertTrue(semantic_qa.normalize_verdict(raw, check_location=False)["passed"])

    def test_string_booleans_and_out_of_range_scores(self):
        verdict = semantic_qa.normalize_verdict(
            {"matches_shot": "true", "subject_consistent": "9", "location_plausible": "false"}
        )
        self.assertTrue(verdict["matches_shot"])
        self.assertEqual(verdict["subject_consistent"], 5.0, "评分夹到 1–5")
        self.assertFalse(verdict["location_plausible"])


class InstructionTests(unittest.TestCase):
    def test_instruction_contains_fields_and_place(self):
        text = semantic_qa.build_shot_instruction(
            order=2, visual_prompt="断桥晨雾", narration="晨雾没散",
            city="杭州", landmark="西湖", subject_lock="同一座断桥",
        )
        for needle in ("杭州西湖", "同一座断桥", "matches_shot", "subject_consistent", "violations"):
            self.assertIn(needle, text)

    def test_location_line_skipped_when_disabled(self):
        text = semantic_qa.build_shot_instruction(
            order=1, visual_prompt="p", narration="n", city="杭州", landmark="西湖",
            subject_lock="s", check_location=False,
        )
        self.assertIn("固定填 true", text)

    def test_regeneration_clause_mentions_specific_problems(self):
        verdict = semantic_qa.normalize_verdict(
            {"matches_shot": False, "subject_consistent": 2,
             "violations": ["modern_intrusion"], "location_plausible": False,
             "note": "出现现代广告牌"}
        )
        clause = semantic_qa.regeneration_clause(verdict)
        self.assertIn("严禁出现现代物闯入", clause)
        self.assertIn("直接呈现本镜描述", clause)
        self.assertIn("外形、材质与比例不变", clause)
        self.assertIn("不要出现其他城市的标志物", clause)
        self.assertIn("出现现代广告牌", clause)


class ReportTests(unittest.TestCase):
    def test_report_passed_and_blocking(self):
        shots = [
            {"shot_id": "shot_1", "verdict": {"passed": True}},
            {"shot_id": "shot_2", "verdict": {"passed": False}},
        ]
        report = semantic_qa.build_report(status="checked", model="m", shots=shots)
        self.assertFalse(report["passed"])
        self.assertEqual(report["blocking_shots"], ["shot_2"])
        self.assertTrue(report["manual_review_required"])

    def test_unconfigured_never_passes(self):
        report = semantic_qa.build_report(status="unconfigured", reason="no model")
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "unconfigured")

    def test_merge_into_visual_qa(self):
        merged = semantic_qa.merge_into_visual_qa({"version": "visual-acceptance-v1"}, {"status": "skipped"})
        self.assertEqual(merged["version"], "visual-acceptance-v1")
        self.assertEqual(merged["semantic"]["status"], "skipped")


class VisionProviderTests(unittest.TestCase):
    def test_unconfigured_returns_none(self):
        self.assertIsNone(build_vision_provider(Settings(vision_model="")))

    def test_configured_builds_provider_with_llm_fallback(self):
        provider = build_vision_provider(
            Settings(
                vision_model="glm-4v-flash",
                vision_base_url="",
                vision_api_key="",
                llm_base_url="https://llm.example/v1",
                llm_api_key="sk-test",
            )
        )
        self.assertIsNotNone(provider)
        self.assertEqual(provider.model, "glm-4v-flash")
        self.assertEqual(provider.base_url, "https://llm.example/v1")
        self.assertEqual(provider.api_key, "sk-test")

    def test_missing_model_returns_none_even_with_base(self):
        self.assertIsNone(
            build_vision_provider(
                Settings(vision_model="", llm_base_url="https://llm.example/v1", llm_api_key="sk")
            )
        )

    def test_build_message_shape(self):
        messages = build_vision_message("判定", ["data:image/png;base64,AAA"])
        self.assertEqual(messages[0]["role"], "user")
        parts = messages[0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "判定"})
        self.assertEqual(parts[1]["type"], "image_url")

    def test_parse_json_object_strips_fence(self):
        self.assertEqual(parse_json_object('```json\n{"ok":true}\n```'), {"ok": True})
        with self.assertRaises(ValueError):
            parse_json_object("[1,2]")

    def test_analyze_rejects_missing_frame(self):
        provider = VisionProvider("https://x", "k", "m")
        with self.assertRaises(RuntimeError):
            provider.analyze([Path("no-such-frame.jpg")], "判定")


class StockLocationTests(unittest.TestCase):
    def test_verify_location_marks_rejected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"\xff\xd8\xff\xd9")

            class _Verdict:
                def analyze(self, frames, instruction, max_tokens=200):
                    self.seen = instruction
                    return {"location_plausible": False, "note": "画面是海边的礁石，不是西湖"}

            vision = _Verdict()
            provider = StockProvider([("pexels", "k", "https://api.pexels.com/v1")], vision=vision)
            status, note = provider._verify_location(frame, "杭州", "西湖")
            self.assertEqual(status, "rejected")
            self.assertIn("西湖", vision.seen)
            self.assertIn("礁石", note)

    def test_verify_location_without_vision_is_unconfigured(self):
        provider = StockProvider([("pexels", "k", "https://api.pexels.com/v1")], vision=None)
        self.assertIsNone(provider.vision)


class _StubVision:
    name = "stub-vision"
    model = "stub-vlm"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def analyze(self, frames, instruction, max_tokens=700):
        self.calls.append(instruction)
        return self.verdicts.pop(0) if self.verdicts else {"matches_shot": True, "subject_consistent": 5}


class SemanticReviewWiringTests(unittest.TestCase):
    def _settings(self, tmp: Path, **overrides) -> Settings:
        base = dict(
            data_dir=tmp,
            semantic_qa_enabled=True,
            semantic_regen_attempts=1,
            semantic_min_subject_score=3,
            semantic_max_frames=1,
            vision_model="",
            vision_base_url="",
            vision_api_key="",
            llm_base_url="",
            llm_api_key="",
        )
        base.update(overrides)
        return Settings(**base)

    def _task(self, tmp: Path) -> Task:
        image = tmp / "frame.jpg"
        image.write_bytes(b"\xff\xd8\xff\xd9")
        task = Task(
            id="task_sem",
            input=TaskInput.from_dict(
                {"city": "杭州", "landmark": "西湖", "theme": "测试", "duration": 15,
                 "asset_source": "ai-video", "shot_count": 2}
            ),
        )
        task.trace = TraceRecord(provider="ai", model="m", prompt_version="v", fallback_used=False)
        task.shots = [
            Shot(
                id="shot_1", order=1, duration_sec=5, title="断桥",
                visual_prompt="断桥晨雾", narration="晨雾没散", subtitle="晨雾没散",
                asset_local_path=str(image), asset_seed=100, asset_media_type="image",
            )
        ]
        task.assets = [
            AssetRecord(
                id="asset_1", task_id=task.id, shot_id="shot_1", name="断桥",
                url="/media/x.jpg", source="ai-video", license="", author="",
                usage_scope="", is_ai_generated=True, local_path=str(image),
                visual_fingerprint="fp", seed=100,
            )
        ]
        return task

    def test_unconfigured_report_without_vision(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            settings = self._settings(tmp)
            orchestrator = Orchestrator(settings, Storage(settings.db_path))
            report = orchestrator._semantic_review(
                self._task(tmp), tmp / "media" / "assets", {}, {}, ("ai-video",), True
            )
            self.assertEqual(report["status"], "unconfigured")
            self.assertFalse(report["passed"])

    def test_checked_report_and_regeneration_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            settings = self._settings(tmp, vision_model="stub-vlm", semantic_regen_attempts=1)
            orchestrator = Orchestrator(settings, Storage(settings.db_path))
            task = self._task(tmp)
            vision = _StubVision([
                {"matches_shot": False, "subject_consistent": 2,
                 "violations": ["modern_intrusion"], "note": "有现代广告牌"},
            ])
            failing = {
                "matches_shot": False, "subject_consistent": 2.0, "reference_similarity": 0.0,
                "violations": ["modern_intrusion"], "blocking_violations": ["modern_intrusion"],
                "warnings": [], "location_plausible": True, "note": "有现代广告牌",
                "passed": False, "reasons": ["画面与分镜描述不符"],
            }
            with patch(
                "backend.app.services.orchestrator.build_vision_provider", return_value=vision
            ), patch(
                "backend.app.services.semantic_qa.sample_frames",
                return_value=[Path(task.shots[0].asset_local_path)],
            ), patch(
                "backend.app.services.semantic_qa.judge_shot", return_value=failing
            ), patch.object(
                Orchestrator,
                "_generate_shot",
                return_value={"shot_id": "shot_1", "order": 1, "asset": None,
                              "used_sources": [], "events": [], "error": "stub-fail"},
            ) as regen:
                report = orchestrator._semantic_review(
                    task, tmp / "media" / "assets", {}, {}, ("ai-video",), True
                )
            self.assertEqual(report["status"], "checked")
            self.assertEqual(report["blocking_shots"], ["shot_1"])
            self.assertEqual(len(report["regenerated"]), 1, "应尝试一次自动重生成")
            self.assertFalse(report["regenerated"][0]["ok"])
            regen.assert_called_once()
            # 重生成失败后，素材指纹被改写，下次重试不会复用被判失败的素材
            self.assertTrue(task.assets[0].visual_fingerprint.startswith("semantic-failed:"))

    def test_disabled_returns_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            settings = self._settings(tmp, semantic_qa_enabled=False)
            orchestrator = Orchestrator(settings, Storage(settings.db_path))
            report = orchestrator._semantic_review(
                self._task(tmp), tmp / "media" / "assets", {}, {}, ("ai-video",), True
            )
            self.assertEqual(report["status"], "skipped")

    def test_stock_source_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            settings = self._settings(tmp, vision_model="stub-vlm", semantic_regen_attempts=2)
            orchestrator = Orchestrator(settings, Storage(settings.db_path))
            task = self._task(tmp)
            failing = {
                "matches_shot": True, "subject_consistent": 5.0, "reference_similarity": 0.0,
                "violations": [], "blocking_violations": [], "warnings": [],
                "location_plausible": False, "note": "", "passed": False,
                "reasons": ["画面地点特征与目标城市/地标不符"],
            }
            with patch(
                "backend.app.services.orchestrator.build_vision_provider", return_value=_StubVision([])
            ), patch(
                "backend.app.services.semantic_qa.sample_frames",
                return_value=[Path(task.shots[0].asset_local_path)],
            ), patch(
                "backend.app.services.semantic_qa.judge_shot", return_value=failing
            ), patch.object(Orchestrator, "_generate_shot") as regen:
                report = orchestrator._semantic_review(
                    task, tmp / "media" / "assets", {}, {}, ("stock",), False
                )
            regen.assert_not_called()
            self.assertEqual(report["blocking_shots"], ["shot_1"])


if __name__ == "__main__":
    unittest.main()
