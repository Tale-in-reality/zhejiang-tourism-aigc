"""旁白文风约束与画面 prompt 精修的单元测试（阶段一文本质量）。

覆盖三层：
1. 词表扫描（copy_quality）：词/句式命中、脚本聚合、去重；
2. 五要素门（consistency.visual_prompt_elements）：完备/缺项；
3. llm.py 的两条改写回路与精修校验：用桩替换网络层，验证循环与事件。
"""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from backend.app.config import Settings
from backend.app.models import Shot, Task, TaskInput, TraceRecord
from backend.app.providers.llm import (
    COPY_STYLE_RULES,
    FEWSHOT_EXAMPLES,
    STYLE_VOICES,
    OpenAICompatibleLLMProvider,
    _refine_instruction,
    plan_narration_budget,
)
from backend.app.services import copy_quality
from backend.app.services.consistency import (
    missing_visual_prompt_elements,
    visual_prompt_elements,
)
from backend.app.services.orchestrator import PIPELINE, Orchestrator
from backend.app.storage import Storage


class CopyScanTests(unittest.TestCase):
    def test_banned_words_and_patterns_hit(self):
        hits = copy_quality.scan_text("走进绍兴，仿佛走进了一幅水墨画卷，让我们一起邂逅绝美时光。")
        terms = copy_quality.hit_terms(hits)
        self.assertIn("绝美", terms)
        self.assertIn("邂逅", terms)
        self.assertIn("让我们一起", terms)
        pattern_terms = [hit["term"] for hit in hits if hit["label"] != "AI 腔词"]
        self.assertTrue(any("水墨画" in term for term in pattern_terms), pattern_terms)

    def test_clean_text_passes(self):
        self.assertEqual(
            copy_quality.scan_text("晨雾还没散，断桥的石拱先醒了。石栏上的雨珠滑下去。"),
            [],
        )

    def test_scan_script_aggregates_fields(self):
        script = {
            "title": "绝美西湖",
            "summary": "",
            "cta": "",
            "shots": [
                {
                    "id": "shot_1",
                    "narration": "让我们一起走进西湖。",
                    "subtitle": "让我们一起走进西湖",
                    "visual_prompt": "湖面如画",
                },
            ],
        }
        scan = copy_quality.scan_script(script)
        self.assertFalse(scan["ok"])
        fields = {hit["field"] for hit in scan["hits"]}
        self.assertIn("title", fields)
        self.assertIn("narration", fields)
        self.assertIn("subtitle", fields)
        # visual_prompt 的命中只记录，不触发改写
        self.assertEqual(len(scan["prompt_hits"]), 0)

    def test_banned_terms_file_parses(self):
        terms = copy_quality.load_banned_terms()
        self.assertTrue(terms["words"], "词表不能为空")
        self.assertTrue(terms["patterns"], "句式正则不能为空")
        for entry in terms["patterns"]:
            self.assertIn("regex", entry)

    def test_hit_fields_format(self):
        hits = [{"field": "narration", "term": "绝美", "excerpt": "绝美西湖"}]
        text = copy_quality.format_hit_fields(hits)
        self.assertIn("narration", text)
        self.assertIn("绝美西湖", text)


class FiveElementTests(unittest.TestCase):
    RICH_PROMPT = (
        "主体：青灰色石拱断桥，石面有凿痕；环境与光线：清晨薄雾，散射光；"
        "构图与景别：中景平视，桥栏占下半；色彩：冷灰调，主色青灰；"
        "画面内动态：晨雾贴湖面流动。"
    )

    def test_complete_prompt_passes_all(self):
        elements = visual_prompt_elements(self.RICH_PROMPT)
        self.assertTrue(all(elements.values()), elements)
        self.assertEqual(missing_visual_prompt_elements(self.RICH_PROMPT), [])

    def test_bare_prompt_missing_most(self):
        missing = missing_visual_prompt_elements("西湖断桥")
        self.assertIn("环境与光线（时段/天气/光源方向）", missing)
        self.assertIn("构图与景别（景别/机位/前后景）", missing)
        self.assertIn("画面内动态", missing)

    def test_empty_prompt_only_subject_false(self):
        elements = visual_prompt_elements("")
        self.assertFalse(elements["subject"])


class StyleRulesTests(unittest.TestCase):
    def test_style_voices_cover_frontend_options(self):
        for style in ("诗意纪实", "城市漫游", "国风雅韵", "青春活力", "节庆热烈"):
            self.assertIn(style, STYLE_VOICES)

    def test_rules_and_fewshot_content(self):
        self.assertIn("绝美", COPY_STYLE_RULES)
        self.assertIn("让我们一起", COPY_STYLE_RULES)
        self.assertIn("反例", FEWSHOT_EXAMPLES)
        self.assertIn("正例", FEWSHOT_EXAMPLES)


class _StubChatProvider(OpenAICompatibleLLMProvider):
    """替换网络层：按脚本顺序返回预置 JSON；同时记录调用次数。"""

    def __init__(self, responses: list[str]) -> None:
        super().__init__("https://stub", "key", "stub-model")
        self.responses = list(responses)
        self.calls = 0

    def _complete(self, messages):
        self.calls += 1
        return self.responses.pop(0)


def _script_payload(narrations, visual="湖面，竖屏构图"):
    return {
        "title": "西湖春日",
        "summary": "三十秒走过西湖。",
        "cta": "来西湖走一走。",
        "tags": ["西湖"],
        "narration": "".join(narrations),
        "knowledge": {"fact": "", "color": "#24526a", "accent": "#d6ad60", "aliases": []},
        "shots": [
            {
                "id": f"shot_{index}",
                "order": index,
                "duration_sec": 5,
                "title": f"镜头{index}",
                "visual_prompt": visual,
                "narration": text,
                "subtitle": text,
                "transition": "fade",
            }
            for index, text in enumerate(narrations, start=1)
        ],
    }


class CopyRewriteLoopTests(unittest.TestCase):
    def test_hit_triggers_rewrite_and_clean_result(self):
        # 第一版：字数达标但命中词表；第二版：干净且字数不变。
        budget = plan_narration_budget(30, 2)
        target = int(budget["per_shot_chars"])
        dirty = _script_payload(["绝美的西湖" + "风" * (target - 5), "苏堤春晓" + "柳" * (target - 4)])
        clean = _script_payload([("晨雾没散，" + "桥" * (target - 5)), ("苏堤柳色" + "柳" * (target - 4))])
        provider = _StubChatProvider(
            [json.dumps(dirty, ensure_ascii=False), json.dumps(clean, ensure_ascii=False)]
        )
        result = provider.generate("生成脚本", {"duration": 30, "shot_count": 2, "city": "杭州"})
        self.assertEqual(provider.calls, 2, "词表命中应触发一轮改写")
        scan = copy_quality.scan_script(result.data)
        self.assertTrue(scan["ok"])
        self.assertTrue(result.event["copy_scan"]["ok"])
        self.assertEqual(result.event["copy_scan"]["rewrite_attempts"], 1)
        self.assertEqual(result.event["prompt_version"], "zj-tourism-prompt-v3")

    def test_clean_first_pass_no_rewrite(self):
        budget = plan_narration_budget(30, 2)
        target = int(budget["per_shot_chars"])
        clean = _script_payload([("晨雾没散，" + "桥" * (target - 5)), ("苏堤柳色" + "柳" * (target - 4))])
        provider = _StubChatProvider([json.dumps(clean, ensure_ascii=False)])
        result = provider.generate("生成脚本", {"duration": 30, "shot_count": 2, "city": "杭州"})
        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.event["copy_scan"]["ok"])


class RefineLoopTests(unittest.TestCase):
    def test_validated_refined_passes(self):
        provider = _StubChatProvider([])
        payload = {
            "shots": [
                {
                    "id": "shot_1",
                    "visual_prompt": "完整 prompt",
                    "search_queries": ["West Lake", "主查询", "备选1", "多余"],
                },
            ]
        }
        refined = provider._validated_refined(payload, [{"id": "shot_1"}])
        self.assertEqual(refined[0]["visual_prompt"], "完整 prompt")
        self.assertEqual(len(refined[0]["search_queries"]), 3, "search_queries 截到 3 条")

    def test_validated_refined_rejects_mismatch(self):
        provider = _StubChatProvider([])
        with self.assertRaises(RuntimeError):
            provider._validated_refined({"shots": []}, [{"id": "shot_1"}])
        with self.assertRaises(RuntimeError):
            provider._validated_refined(
                {"shots": [{"id": "shot_x", "visual_prompt": "p"}]}, [{"id": "shot_1"}]
            )

    def test_refine_retry_on_missing_elements(self):
        provider = _StubChatProvider([])
        # 画面里有光线与动态，但没有构图与色彩 → 第一轮不过，打回后仍不过，
        # 事件如实记录 pass_rate < 1，不抛错（精修是增强不是关卡）。
        provider._complete = lambda messages: json.dumps(
            {
                "shots": [
                    {
                        "id": "shot_1",
                        "visual_prompt": "西湖断桥，晨雾贴着湖面流动，桥面石栏有水痕。",
                        "search_queries": ["West Lake Hangzhou"],
                    }
                ]
            },
            ensure_ascii=False,
        )
        outcome = provider.refine_prompts(
            [{"id": "shot_1", "visual_prompt": "断桥", "narration": "晨雾"}],
            {"city": "杭州", "landmark": "西湖", "style": "诗意纪实"},
        )
        self.assertEqual(outcome["event"]["refined"], 1)
        self.assertFalse(outcome["event"]["ok"])
        self.assertLess(outcome["event"]["five_element_pass_rate"], 1.0)
        self.assertEqual(outcome["event"]["element_rewrite_attempts"], 1)

    def test_refine_instruction_carries_lock_and_elements(self):
        text = _refine_instruction(
            [{"id": "shot_1", "visual_prompt": "断桥", "narration": "n"}],
            "诗意纪实",
            "同一座断桥",
        )
        self.assertIn("主体锁定：同一座断桥", text)
        self.assertIn("五要素", text)
        self.assertIn("search_queries", text)


class PipelineWiringTests(unittest.TestCase):
    def test_prompt_refine_stage_positioned(self):
        names = [stage.name for stage in PIPELINE]
        self.assertIn("prompt_refine", names)
        self.assertEqual(names.index("prompt_refine"), names.index("scripting") + 1)
        progresses = [(stage.started_progress, stage.completed_progress) for stage in PIPELINE]
        self.assertEqual(progresses, sorted(progresses, key=lambda item: item[0]))

    def test_offline_source_skips_refine(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp))
            storage = Storage(settings.db_path)
            orchestrator = Orchestrator(settings, storage)
            task = Task(
                id="task_refine_skip",
                input=TaskInput.from_dict(
                    {
                        "city": "杭州",
                        "theme": "测试",
                        "duration": 15,
                        "asset_source": "offline",
                        "shot_count": 2,
                    }
                ),
            )
            task.trace = TraceRecord(provider="offline", model="", prompt_version="", fallback_used=False)
            task.shots = [
                Shot(id="shot_1", order=1, duration_sec=5, title="t", visual_prompt="p", narration="n", subtitle="n"),
                Shot(id="shot_2", order=2, duration_sec=5, title="t", visual_prompt="p", narration="n", subtitle="n"),
            ]
            orchestrator._prompt_refine(task)
            self.assertEqual(task.shots[0].visual_prompt, "p", "离线来源不应改写 prompt")
            self.assertEqual(task.shots[0].search_queries, [])
            skipped = [
                event
                for event in task.trace.events
                if event.get("stage") == "prompt_refine" and event.get("status") == "skipped"
            ]
            self.assertTrue(skipped)


class ShotRoundtripTests(unittest.TestCase):
    def test_search_queries_survive_roundtrip(self):
        shot = Shot(
            id="shot_1", order=1, duration_sec=5, title="t",
            visual_prompt="p", narration="n", subtitle="n",
            search_queries=["West Lake Hangzhou", "Broken Bridge"],
        )
        restored = Shot(**asdict(shot))
        self.assertEqual(restored.search_queries, ["West Lake Hangzhou", "Broken Bridge"])

    def test_legacy_task_dict_without_field(self):
        # 旧任务 JSON 没有 search_queries 字段，Shot(**shot) 仍可反序列化
        data = asdict(
            Shot(
                id="shot_1", order=1, duration_sec=5, title="t",
                visual_prompt="p", narration="n", subtitle="n",
            )
        )
        data.pop("search_queries")
        restored = Shot(**data)
        self.assertEqual(restored.search_queries, [])


if __name__ == "__main__":
    unittest.main()
