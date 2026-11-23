"""候选池 → 选择 → 详述（explore 模式）的单元与端到端测试。

覆盖：
1. 候选校验门（配额/方向互斥/锚点可拍/地点归属）；
2. 选择校验（硬错误 vs 提醒）；
3. 时长分配与详述校验门（字段/锚点继承/长度/自洽/禁词）；
4. 两条重写回路（候选整体重写、详述只回炉出错镜头）；
5. orchestrator 全流程：create_candidates → select → generate_detail。
"""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from backend.app.config import Settings
from backend.app.models import Shot, ShotCandidate, TaskInput
from backend.app.providers.llm import OpenAICompatibleLLMProvider, plan_narration_budget
from backend.app.providers.offline import OfflineLLMProvider
from backend.app.services import shot_candidates as sc
from backend.app.services import shot_detail as sd
from backend.app.services.orchestrator import Orchestrator
from backend.app.storage import Storage


def _candidate(index, role, *, title=None, place="杭州西湖", time="清晨", size="中景", imagery=None):
    payload = {
        "id": f"c{index}",
        "role": role,
        "title": title or f"西湖{size}{index}",
        "hook": f"{time}的西湖，湖面与石栏",
        "imagery": imagery or [f"石栏缺口{index}", f"湖面雾{index}"],
        "place": place,
        "time_of_day": time,
        "shot_size": size,
        "motion": "雾气缓慢流动",
        "why": "建立辨识度",
        "duration_hint": 5,
    }
    return payload


def _valid_candidates():
    roles = (
        ["opening"] * 2 + ["subject"] * 4 + ["detail"] * 2 + ["person"] * 1 + ["closing"] * 1
    )
    sizes = ["远景", "全景", "中景", "近景", "特写"]
    times = ["清晨", "正午", "黄昏", "夜"]
    return [
        _candidate(
            index, role,
            title=f"西湖{role}{index}",
            time=times[(index - 1) % 4],
            size=sizes[(index - 1) % 5],
        )
        for index, role in enumerate(roles, start=1)
    ]


def _detail_item(candidate_id, *, anchors, narration_len=29, body_pad="", time="清晨"):
    body = (
        "青灰石拱与石栏，栏内侧缺口与雨痕清楚可见；"
        "清晨薄雾，低角度散射光从右上方来，明暗过渡平缓；"
        "中景平视，主体位于画面中部，上方留白给字幕；"
        "冷灰色调，青灰与雾白为主；"
        "雾气缓慢流动，水面反光轻轻晃动，画面稳定。"
    ) + body_pad
    while len(body) < 115:
        body += "石面水痕与远处山影层次清楚。"
    narration = ("晨雾没散，断桥先醒。" + "石栏上的雨珠滑下去，桥面留着水痕。")
    narration = (narration * 2)[:narration_len]
    return {
        "candidate_id": candidate_id,
        "subject": {"name": "断桥石拱", "material": "青灰石材", "details": ["石栏缺口", "雨痕"], "scale": "中景"},
        "environment": {"place": "杭州西湖断桥", "season": "春", "weather": "薄雾", "time": time},
        "lighting": {"direction": "右上方", "quality": "低角度散射光", "contrast": "低"},
        "composition": {"shot_size": "中景", "angle": "平视", "framing": "主体居中，前景留环境", "negative_space": "上方留白"},
        "palette": {"temperature": "冷", "dominant": ["青灰", "雾白"]},
        "motion": {"in_frame": ["雾气缓慢流动"], "camera": "静止"},
        "must_include": list(anchors),
        "must_avoid": ["文字", "现代广告牌"],
        "continuity": {"same_subject": True, "same_palette": True, "relation_to_prev": "延续上一镜色调"},
        "visual_prompt": body,
        "narration": narration,
        "subtitle": narration,
    }


def _plan(shot_count=4):
    total, roles = sc.candidate_plan(sc.target_shot_count(30, shot_count))
    return total, roles


class CandidatePlanTests(unittest.TestCase):
    def test_candidate_total_scales_with_shot_count(self):
        total4, roles4 = _plan(4)
        self.assertEqual(total4, 10, "4 镜 ×2.5 = 10，保持原行为")
        self.assertEqual(sum(roles4.values()), total4)
        total8, roles8 = _plan(8)
        self.assertEqual(total8, 20)
        self.assertEqual(sum(roles8.values()), total8)
        total2, roles2 = _plan(2)
        self.assertEqual(total2, 5)
        self.assertEqual(sum(roles2.values()), total2)

    def test_auto_shot_count_from_duration(self):
        self.assertEqual(sc.target_shot_count(15, 0), 3)
        self.assertEqual(sc.target_shot_count(30, 0), 6)
        self.assertEqual(sc.target_shot_count(60, 0), 12)
        self.assertEqual(sc.target_shot_count(30, 8), 8, "显式镜头数优先")

    def test_plan_is_capped(self):
        total, roles = sc.candidate_plan(12)
        self.assertEqual(total, sc.MAX_CANDIDATES)
        self.assertEqual(sum(roles.values()), sc.MAX_CANDIDATES)


class CandidateValidationTests(unittest.TestCase):
    def test_valid_candidates_pass(self):
        total, roles = _plan(4)
        _, candidates = sc.normalize_candidates({"candidates": _valid_candidates()})
        self.assertEqual(len(candidates), total)
        self.assertEqual(
            sc.validate_candidates(candidates, city="杭州", expected_total=total, roles=roles), []
        )

    def test_wrong_quota_fails(self):
        total, roles = _plan(4)
        _, candidates = sc.normalize_candidates({"candidates": _valid_candidates()[:9]})
        problems = sc.validate_candidates(
            candidates, city="杭州", expected_total=total, roles=roles
        )
        self.assertTrue(any(f"必须正好 {total}" in item for item in problems))

    def test_duplicate_direction_fails(self):
        total, roles = _plan(4)
        payload = _valid_candidates()
        payload[1] = _candidate(2, "opening", title=payload[0]["title"], time=payload[0]["time_of_day"], size=payload[0]["shot_size"])
        _, candidates = sc.normalize_candidates({"candidates": payload})
        problems = sc.validate_candidates(
            candidates, city="杭州", expected_total=total, roles=roles
        )
        self.assertTrue(any("方向重复" in item for item in problems), problems)

    def test_unshootable_anchor_fails(self):
        total, roles = _plan(4)
        payload = _valid_candidates()
        payload[0]["imagery"] = ["绝美氛围感", "美景"]
        _, candidates = sc.normalize_candidates({"candidates": payload})
        problems = sc.validate_candidates(
            candidates, city="杭州", expected_total=total, roles=roles
        )
        self.assertTrue(any("不可拍摄" in item for item in problems), problems)

    def test_place_outside_city_fails(self):
        total, roles = _plan(4)
        payload = _valid_candidates()
        payload[0]["place"] = "苏州园林"
        _, candidates = sc.normalize_candidates({"candidates": payload})
        problems = sc.validate_candidates(
            candidates, city="杭州", expected_total=total, roles=roles
        )
        self.assertTrue(any("不在所选城市" in item for item in problems), problems)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        _, self.candidates = sc.normalize_candidates({"candidates": _valid_candidates()})

    def test_selection_sorted_and_warned(self):
        selection = [
            {"candidate_id": "c3", "order": 2},
            {"candidate_id": "c1", "order": 1},
        ]
        ordered, warnings = sc.select_candidates(self.candidates, selection)
        self.assertEqual([row["candidate_id"] for row in ordered], ["c1", "c3"])
        self.assertEqual([row["order"] for row in ordered], [1, 2])
        self.assertTrue(warnings, "缺收束/细节时应给提醒")

    def test_unknown_candidate_rejected(self):
        with self.assertRaises(ValueError):
            sc.select_candidates(self.candidates, [{"candidate_id": "cx", "order": 1}])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            sc.select_candidates(
                self.candidates,
                [{"candidate_id": "c1", "order": 1}, {"candidate_id": "c1", "order": 2}],
            )

    def test_too_many_rejected(self):
        selection = [{"candidate_id": f"c{i}", "order": i} for i in range(1, 14)]
        with self.assertRaises(ValueError):
            sc.select_candidates(self.candidates, selection)


class DurationTests(unittest.TestCase):
    def test_durations_sum_to_total(self):
        durations = sd.plan_detail_durations([5, 5, 4, 6], 30)
        self.assertAlmostEqual(sum(durations), 30.0, places=2)
        self.assertTrue(all(value >= 1 for value in durations))

    def test_impossible_split_raises(self):
        with self.assertRaises(ValueError):
            sd.plan_detail_durations([8] * 12, 10)


class DetailGateTests(unittest.TestCase):
    def _selected(self):
        return [
            {"candidate_id": f"c{i}", "imagery": [f"锚点A{i}", f"锚点B{i}"], "place": "杭州西湖"}
            for i in range(1, 5)
        ]

    def _budget(self):
        return plan_narration_budget(30, 4)

    def test_valid_detail_passes(self):
        selected = self._selected()
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"锚点B{i}", f"补充{i}"])
            for i in range(1, 5)
        ]
        normalized = sd.normalize_detail({"shots": shots}, selected=selected, per_shot_seconds=[7.5] * 4)
        gate = sd.validate_detail(
            normalized, selected=selected, city="杭州", narration_budget=self._budget(),
            composed=sd.build_composed_prompts(normalized, {"city": "杭州", "landmark": "西湖"}),
        )
        self.assertTrue(gate.ok, gate.problems)

    def test_missing_inherited_anchor_fails(self):
        selected = self._selected()
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"补充{i}", f"补充2{i}"])
            for i in range(1, 5)
        ]
        normalized = sd.normalize_detail({"shots": shots}, selected=selected, per_shot_seconds=[7.5] * 4)
        gate = sd.validate_detail(normalized, selected=selected, city="杭州", narration_budget=self._budget())
        self.assertFalse(gate.ok)
        self.assertTrue(any("未继承候选锚点" in item for item in gate.problems), gate.problems)

    def test_short_body_fails(self):
        selected = self._selected()
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"锚点B{i}", f"补充{i}"], narration_len=29)
            for i in range(1, 5)
        ]
        shots[0]["visual_prompt"] = "西湖断桥，晨雾。"
        normalized = sd.normalize_detail({"shots": shots}, selected=selected, per_shot_seconds=[7.5] * 4)
        gate = sd.validate_detail(normalized, selected=selected, city="杭州", narration_budget=self._budget())
        self.assertFalse(gate.ok)
        self.assertTrue(any("低于" in item and "字" in item for item in gate.problems), gate.problems)

    def test_time_light_conflict_fails(self):
        selected = self._selected()
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"锚点B{i}", f"补充{i}"], time="夜")
            for i in range(1, 5)
        ]
        shots[0]["lighting"] = {"direction": "正上方", "quality": "正午顶光", "contrast": "高"}
        normalized = sd.normalize_detail({"shots": shots}, selected=selected, per_shot_seconds=[7.5] * 4)
        gate = sd.validate_detail(normalized, selected=selected, city="杭州", narration_budget=self._budget())
        self.assertTrue(any("矛盾" in item for item in gate.problems), gate.problems)

    def test_banned_words_fail(self):
        selected = self._selected()
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"锚点B{i}", f"补充{i}"])
            for i in range(1, 5)
        ]
        shots[1]["narration"] = "绝美西湖，让我们一起邂逅。" + shots[1]["narration"]
        shots[1]["subtitle"] = shots[1]["narration"]
        normalized = sd.normalize_detail({"shots": shots}, selected=selected, per_shot_seconds=[7.5] * 4)
        gate = sd.validate_detail(normalized, selected=selected, city="杭州", narration_budget=self._budget())
        self.assertTrue(any("AI 腔词表" in item for item in gate.problems), gate.problems)


class _StubChat(OpenAICompatibleLLMProvider):
    def __init__(self, responses):
        super().__init__("https://stub", "k", "m")
        self.responses = list(responses)
        self.calls = 0

    def _complete(self, messages):
        self.calls += 1
        return self.responses.pop(0)


class RewriteLoopTests(unittest.TestCase):
    def test_candidates_rewrite_after_quota_failure(self):
        bad = {"arc": {}, "candidates": _valid_candidates()[:9]}
        good = {"arc": {"opening": "a"}, "candidates": _valid_candidates()}
        provider = _StubChat([json.dumps(bad, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
        result = provider.candidates({"city": "杭州", "theme": "春游", "duration": 30, "shot_count": 4})
        self.assertEqual(provider.calls, 2, "候选校验失败应触发一轮重写")
        self.assertEqual(len(result.data["candidates"]), 10)
        self.assertEqual(result.event["rewrite_attempts"], 1)

    def test_detail_selective_rewrite_merges(self):
        selected = [
            {"candidate_id": f"c{i}", "imagery": [f"锚点A{i}", f"锚点B{i}"], "place": "杭州西湖"}
            for i in range(1, 5)
        ]
        budget = plan_narration_budget(30, 4)
        shots = [
            _detail_item(f"c{i}", anchors=[f"锚点A{i}", f"锚点B{i}", f"补充{i}"])
            for i in range(1, 5)
        ]
        # 第一镜丢一个继承锚点 → 只回炉第一镜
        shots[0]["must_include"] = ["锚点A1", "补充1", "补充2"]
        first = {"title": "t", "summary": "s", "cta": "c", "shots": shots}
        fixed_shot = _detail_item("c1", anchors=["锚点A1", "锚点B1", "补充1"])
        provider = _StubChat([
            json.dumps(first, ensure_ascii=False),
            json.dumps({"shots": [fixed_shot]}, ensure_ascii=False),
        ])
        outcome = provider.detail({
            "selected": selected,
            "per_shot_seconds": [7.5] * 4,
            "task_input": {"city": "杭州", "landmark": "西湖", "duration": 30, "theme": "春游"},
            "style_ctx": {"city": "杭州", "landmark": "西湖"},
        })
        self.assertEqual(provider.calls, 2)
        self.assertTrue(outcome["gate"]["ok"], outcome["gate"]["problems"])
        self.assertEqual([shot["candidate_id"] for shot in outcome["shots"]], ["c1", "c2", "c3", "c4"])
        self.assertEqual(outcome["event"]["rewrite_attempts"], 1)


class OfflineExploreTests(unittest.TestCase):
    def test_offline_candidates_and_detail_pass_gates(self):
        provider = OfflineLLMProvider()
        context = {
            "city": "杭州", "landmark": "西湖", "theme": "西湖春游", "culture": "宋韵",
            "festival": "", "audience": "年轻游客", "style": "诗意纪实",
            "custom_brief": "", "duration": 30, "shot_count": 4,
        }
        result = provider.candidates(context)
        arc, candidates = sc.normalize_candidates(result.data)
        total, roles = _plan(4)
        self.assertEqual(len(candidates), total)
        self.assertEqual(
            sc.validate_candidates(candidates, city="杭州", expected_total=total, roles=roles), []
        )
        selected_rows = [
            {**candidate.to_dict(), "candidate_id": candidate.id}
            for candidate in candidates
            if candidate.role in {"opening", "subject", "detail", "closing"}
        ][:4]
        durations = sd.plan_detail_durations(
            [row["duration_hint"] for row in selected_rows], 30
        )
        outcome = provider.detail({
            "selected": selected_rows,
            "per_shot_seconds": durations,
            "task_input": context,
            "style_ctx": {"city": "杭州", "landmark": "西湖", "style": "诗意纪实"},
        })
        self.assertTrue(outcome["gate"]["ok"], outcome["gate"]["problems"])
        self.assertEqual(len(outcome["shots"]), 4)
        for shot in outcome["shots"]:
            self.assertGreaterEqual(len(shot["visual_prompt"]), sd.BODY_MIN_CHARS)


class OrchestratorExploreFlowTests(unittest.TestCase):
    def test_full_explore_flow_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                data_dir=Path(directory), default_provider="offline", asset_provider="offline"
            )
            storage = Storage(settings.db_path)
            orchestrator = Orchestrator(settings, storage)
            task_input = TaskInput.from_dict(
                {"city": "杭州", "landmark": "西湖", "theme": "西湖春游", "duration": 30,
                 "style": "诗意纪实", "shot_count": 4, "asset_source": "offline"}
            )
            task = orchestrator.create_candidates(task_input)
            self.assertEqual(task.status, "candidates_ready")
            self.assertEqual(len(task.shot_candidates), 10)
            self.assertTrue(task.candidates_arc)

            chosen = [
                candidate.id for candidate in task.shot_candidates
                if candidate.role in {"opening", "subject", "detail", "closing"}
            ][:4]
            result = orchestrator.select_candidates(
                task.id, {"selection": [{"candidate_id": cid, "order": i} for i, cid in enumerate(chosen, 1)]}
            )
            self.assertEqual(result["status"], "selected")

            detail = orchestrator.generate_detail(task.id)
            self.assertTrue(detail["gate"]["ok"], detail["gate"]["problems"])
            stored = storage.get_task(task.id)
            self.assertEqual(stored["status"], "draft")
            self.assertEqual(len(stored["shots"]), 4)
            self.assertAlmostEqual(
                sum(float(shot["duration_sec"]) for shot in stored["shots"]), 30.0, places=1
            )
            for shot in stored["shots"]:
                self.assertTrue(shot["composed_prompt"], "每镜应保存渲染后的完整 prompt")
                self.assertTrue(shot["intent_anchors"], "每镜应继承候选锚点")
            # 候选池与选择能从存储里还原
            self.assertEqual(len(stored["shot_candidates"]), 10)
            self.assertEqual(len(stored["selection"]), 4)

    def test_detail_before_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory), default_provider="offline")
            storage = Storage(settings.db_path)
            orchestrator = Orchestrator(settings, storage)
            task_input = TaskInput.from_dict(
                {"city": "杭州", "theme": "测试", "duration": 30, "shot_count": 4}
            )
            task = orchestrator.create_candidates(task_input)
            with self.assertRaises(ValueError):
                orchestrator.generate_detail(task.id)


if __name__ == "__main__":
    unittest.main()


class BackNavigationTests(unittest.TestCase):
    """从后续步骤回退修改：允许重选、下游保留并标过期、锚点可编辑。"""

    def _flow(self, settings, orchestrator):
        task_input = TaskInput.from_dict(
            {"city": "杭州", "landmark": "西湖", "theme": "西湖春游", "duration": 30,
             "style": "诗意纪实", "shot_count": 4, "asset_source": "offline"}
        )
        task = orchestrator.create_candidates(task_input)
        chosen = [
            candidate.id for candidate in task.shot_candidates
            if candidate.role in {"opening", "subject", "detail", "closing"}
        ][:4]
        orchestrator.select_candidates(
            task.id,
            {"selection": [{"candidate_id": cid, "order": i} for i, cid in enumerate(chosen, 1)]},
        )
        orchestrator.generate_detail(task.id)
        return task, chosen

    def test_reselection_from_draft_keeps_downstream_and_marks_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                data_dir=Path(directory), default_provider="offline", asset_provider="offline"
            )
            storage = Storage(settings.db_path)
            orchestrator = Orchestrator(settings, storage)
            task, chosen = self._flow(settings, orchestrator)
            shots_before = len(storage.get_task(task.id)["shots"])
            self.assertEqual(shots_before, 4)

            # 草稿态回退重选（换一个候选）
            other = next(
                candidate.id for candidate in task.shot_candidates if candidate.id not in chosen
            )
            result = orchestrator.select_candidates(
                task.id,
                {"selection": [{"candidate_id": cid, "order": i} for i, cid in enumerate([chosen[0], other], 1)]},
            )
            self.assertEqual(result["status"], "selected")
            stored = storage.get_task(task.id)
            self.assertEqual(len(stored["shots"]), 4, "下游分镜保留，不自动删除")
            events = [event for event in (stored.get("trace") or {}).get("events", [])
                      if event.get("type") == "shot_selection"]
            self.assertTrue(events[-1].get("stale_downstream"))

    def test_patch_shot_accepts_anchor_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                data_dir=Path(directory), default_provider="offline", asset_provider="offline"
            )
            storage = Storage(settings.db_path)
            orchestrator = Orchestrator(settings, storage)
            task, chosen = self._flow(settings, orchestrator)
            shot_id = storage.get_task(task.id)["shots"][0]["id"]
            orchestrator.update_shot(
                task.id,
                shot_id,
                {"intent_anchors": "石栏缺口、薄雾，远山", "must_avoid": "文字；现代广告牌"},
            )
            shot = storage.get_task(task.id)["shots"][0]
            self.assertEqual(shot["intent_anchors"], ["石栏缺口", "薄雾", "远山"])
            self.assertEqual(shot["must_avoid"], ["文字", "现代广告牌"])
