from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .contract import TRANSITIONS


from ..services.consistency import missing_visual_prompt_elements
from ..services.copy_quality import format_hit_fields, hit_terms, scan_script
from ..services.cost import provider_call, record_response
from ..services import shot_candidates as candidate_service
from ..services import shot_detail as detail_service

def _resolve_knowledge_context(context: dict[str, Any]) -> str:
    """取本次创作的浙江文旅事实参考块，用于注入生成 prompt。

    知识库是增强而非依赖：语料缺失、路径异常或服务不可用时应静默降级为
    空串（prompt 退回原有的"不得编造"约束），绝不能让它阻断脚本生成。
    """
    try:
        from ..services import knowledge

        # None 不能直接 str()——那会得到字面量 "None" 并被当成城市名参与匹配
        def _field(name: str) -> str:
            value = context.get(name)
            return "" if value is None else str(value)

        return knowledge.build_generation_context(
            city=_field("city"),
            landmark=_field("landmark"),
            theme=_field("theme"),
            culture=_field("culture"),
            festival=_field("festival"),
        )
    except Exception:  # noqa: BLE001 - 知识库失败必须降级，不能阻断主流程
        return ""


# 中文配音实测约 4.2~4.5 字/秒（edge-tts 晓晓、rate=1.0、不含句间停顿）。
# 预算按 4.2 算：带内部空格/顿号的旁白实际会慢到约 4.0 字/秒，超出的部分由
# 配音阶段提速兜住（MAX_RATE_SCALE），比反过来留一大段没人说话的空档好。
# 注意：这个值必须按"裁掉引擎静音之后"的测量来定。旧值 3.8 是按含 0.2~0.6 秒
# 引擎内边距的时长标的，裁剪上线后沿用旧值会让成片尾部空出 2~8 秒。
NARRATION_CHARS_PER_SECOND = 4.2

# 字数兜底的改写轮数。一轮实测不总能落进区间，两轮封顶。
NARRATION_REWRITE_ATTEMPTS = 2

# AI 腔词表命中的改写轮数。命中会触发整段旁白改写；一轮改不干净就如实
# 记录剩余命中（event.copy_scan），不再追加轮次——词表拦截是概率性收敛，
# 不能为它无限制烧文本调用。
COPY_REWRITE_ATTEMPTS = 1

# —— 文风约束（去 AI 味）——
# 旁白是给人听的口播稿。评测口径见 scripts/copy_quality_eval.py；
# 词表见 backend/app/services/copy_banned_terms.json（与拦截口径同一份）。
# 按叙事风格分化语气：前端提供 5 种风格，不知道的风格退回通用默认。
STYLE_VOICES: dict[str, str] = {
    "诗意纪实": "白描短句，克制用形容词；多用名词与动词，允许留白，不堆感叹。",
    "城市漫游": "第二人称带路感，多用方位与脚步动线（巷口、拐角、桥堍）；口语但不玩网梗。",
    "国风雅韵": "可用少量文言痕迹（四字格、雅称），不堆典故不掉书袋，句子仍要一听就懂。",
    "青春活力": "轻快短句，可带一点幽默；不用「家人们」「绝绝子」一类直播梗。",
    "节庆热烈": "写现场感：声音（锣鼓、人声）、气味、动作；节奏紧凑，句子更短。",
}
STYLE_VOICE_DEFAULT = "白描为主，具体名词优先，短句优先；像去过当地的人讲的，不像宣传册印的。"

COPY_STYLE_RULES = (
    "文风硬规则（旁白是给人听的口播稿，AI 腔直接废稿）：\n"
    "1. 禁止使用以下词与句式（评测逐条扫描，命中触发整段改写）："
    "绝美、璀璨、邂逅、解锁、天花板、出片圣地、视觉盛宴、心灵之旅、诗与远方、沉浸式体验；"
    "「让我们一起」「在这里，你可以」「不仅…更是…」「仿佛在诉说」「承载着…记忆」「一幅流动的水墨画」。\n"
    "2. 抽象套话必须换成具体名词：写得出材质、颜色、声音、动作、数字的细节。\n"
    "3. 每镜第一句不超过 12 个字，一句话只说一件事。\n"
    "4. 像去过当地的人讲的，可以有观点和体温，不要靠感叹词凑数。\n"
)

FEWSHOT_EXAMPLES = (
    "文风示例（照正例的水准写；反例是要避免的写法）：\n"
    "正例·诗意纪实（杭州西湖）：「晨雾还没散，断桥先醒了。石栏上昨夜的雨珠顺着青灰色石缝滑下去，"
    "桥面留着一道没扫到的雪线。」\n"
    "正例·城市漫游（绍兴黄酒）：「巷口酒旗先晃过来，酱香跟着人走。老师傅不开口，木耙在酒饭里翻三下，"
    "抬眼看你——意思是，急不得。」\n"
    "反例（不要这样写）：「走进绍兴，仿佛走进了一幅水墨画卷。这里不仅有千年古城的厚重底蕴，"
    "更是黄酒文化的璀璨明珠。让我们一起来邂逅这场微醺之旅吧！」"
    "——问题：「仿佛走进」「不仅…更是…」「璀璨」「邂逅」「让我们」全是套话，通篇没有一个能拍进画面的具体细节。\n"
)

# —— 视觉 prompt 精修（prompt_refine 阶段）——
REFINE_SYSTEM_PROMPT = "你是浙江文旅短视频的摄影指导，只输出合法 JSON。"


def _refine_instruction(
    shots: list[dict[str, Any]], style: str, subject_lock: str
) -> str:
    payload = json.dumps(
        [
            {
                "id": shot.get("id"),
                "order": shot.get("order"),
                "visual_prompt": shot.get("visual_prompt"),
                "narration": shot.get("narration"),
            }
            for shot in shots
        ],
        ensure_ascii=False,
    )
    return (
        "把下面每个镜头的画面描述升级为可直接发给图像/视频生成模型的精确 prompt。\n"
        f"全片叙事风格「{style or '未指定'}」。全片主体锁定：{subject_lock}"
        "——每个镜头的主体必须与锁定一致，只改变机位、景别、动作、时段与光线。\n"
        "每个 visual_prompt 必须包含五要素，缺一不可：\n"
        "1) 主体：是什么 + 材质/外观/状态细节；\n"
        "2) 环境与光线：时段（晨/午/黄昏/夜）、天气、光源方向；\n"
        "3) 构图与景别：远景/全景/中景/近景/特写、机位高度或角度、前后景关系；\n"
        "4) 色彩：色温与主色（全片统一）；\n"
        "5) 画面内动态：画面里在动的东西（人物动作、水面、旗帜、树叶）；"
        "只写画面内运动，不要写镜头运动。\n"
        "同时为每个镜头给出一组英文素材检索查询 search_queries："
        "1 条主查询 + 2 条备选，具体到「地名 + 景观/活动」，"
        "供图库（Pexels/Pixabay）检索使用，不要用单词泛词。\n"
        '只输出 JSON：{"shots":[{"id":"原id","visual_prompt":"...","search_queries":["..."]}]}\n'
        f"逐镜输入：{payload}"
    )

# 模型偶尔会把 JSON 包在 Markdown 代码块里，或者返回截断内容。先剥壳再解析，
# 解析失败时把错误回灌给模型重试一次；仍然失败才让任务失败。
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_CAPTION_NOISE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def _strip_json_fence(content: str) -> str:
    return _JSON_FENCE.sub("", str(content or "").strip())


def _parse_json_object(content: str) -> dict[str, Any]:
    text = _strip_json_fence(content)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("LLM returned a non-object JSON response")
    return result


def _caption_key(value: Any) -> str:
    """Compare narration and subtitle without whitespace or punctuation."""
    return _CAPTION_NOISE.sub("", str(value or "")).casefold()


def plan_narration_budget(duration: int, shot_count: int) -> dict[str, float]:
    """把成片时长拆成每镜的字数预算。

    逐镜配音在镜头之间留一段静音（services.audio_sync.SHOT_GAP_TTS_SEC，
    裁掉引擎内边距后由构建器显式补上），那段时间说不了字。算预算前必须先扣掉，
    否则旁白必然超长，成片锁回用户秒数时把最后一句的结尾裁掉——这正是
    "结尾不完整"的成因。
    """
    try:
        from ..services.audio_sync import SHOT_GAP_TTS_SEC
    except Exception:  # noqa: BLE001 - 取不到就退回同一份默认值
        SHOT_GAP_TTS_SEC = 0.300
    silence_total = SHOT_GAP_TTS_SEC * max(0, shot_count - 1)
    usable = max(1.0, float(duration) - silence_total)
    total_chars = max(shot_count * 6, int(usable * NARRATION_CHARS_PER_SECOND))
    # 向下取整，不用 round：每镜上限回乘镜头数必须仍不超过总预算，
    # 否则 8 镜时 round 出来的上限会反超总预算，旁白又超长被裁。
    per_shot_chars = max(6, total_chars // shot_count)
    return {
        "silence_total": silence_total,
        "per_shot_seconds": usable / shot_count,
        "total_chars": total_chars,
        "per_shot_chars": per_shot_chars,
        # 下限留 2 字余量。旁白偏短和偏长一样是废片：短了成片会剩下没有解说、
        # 只有静止画面的空档（画面锁在用户选的秒数，配音补不满就只能冻最后一帧）。
        "per_shot_floor": max(4, per_shot_chars - 2),
    }


@dataclass
class LLMResult:
    data: dict[str, Any]
    provider: str
    model: str
    fallback_used: bool
    event: dict[str, Any]


class LLMProvider:
    name = "base"

    def generate(self, prompt: str, context: dict[str, Any]) -> LLMResult:
        raise NotImplementedError


class OpenAICompatibleLLMProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 45) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, context: dict[str, Any]) -> LLMResult:
        if not self.base_url or not self.api_key or not self.model:
            raise RuntimeError("LLM requires base_url, api_key, and model")
        duration = int(context.get("duration", 30))
        # 自动镜头数：按每镜约 5 秒切，夹在 2–12 之间。
        # 图生视频单镜 4–15 秒，5 秒是既够运动、又不至于让单镜成本翻倍的中间值。
        # 短时长要额外保护：5 秒切成 2 镜只有 2.5 秒/镜，低于模型下限，
        # 所以镜头数不超过 floor(duration / 4)。
        max_by_floor = max(1, int(duration // 4))
        auto_shots = min(12, max_by_floor, max(2, round(duration / 5)))
        auto_shots = max(1, auto_shots)
        shot_count = int(context.get("shot_count") or auto_shots)
        # 每镜旁白要和它的配音时长对得上：中文配音约每秒 4 字。旁白超出镜头时长时，
        # 渲染器会把成片锁回用户选择的秒数，多出来的部分只能被裁掉。
        budget = plan_narration_budget(duration, shot_count)
        per_shot_seconds = budget["per_shot_seconds"]
        per_shot_chars = budget["per_shot_chars"]
        per_shot_floor = budget["per_shot_floor"]
        total_chars = budget["total_chars"]
        custom_brief = str(context.get("custom_brief", "")).strip()
        subject_lock = str(context.get("subject_lock", "")).strip()
        if not subject_lock:
            place = f"{context.get('city', '')}{context.get('landmark', '')}".strip()
            subject_lock = f"{place or '核心景点'}的可辨识主体、轮廓、材质和标志性细节"
        knowledge_context = _resolve_knowledge_context(context)
        instruction = self._build_instruction(
            prompt,
            context,
            shot_count=shot_count,
            duration=duration,
            budget=budget,
            subject_lock=subject_lock,
            knowledge_context=knowledge_context,
            custom_brief=custom_brief,
        )
        # 参考块拼在 user prompt 中创作要求之后。位置与写法一律以
        # scripts/kb_injection_eval.py 的实测数据为准，不要凭单次观感调整。
        messages = [
            {"role": "system", "content": "你是严谨的浙江文旅视频编导，只输出合法 JSON。"},
            {"role": "user", "content": instruction},
        ]
        result, shots = self._generate_validated(messages, shot_count)

        # 字数靠 prompt 措辞约束不可靠：同一份约束实测出过"每镜 17 字"（结尾被裁）
        # 和"每镜 7 字"（成片后半段没有解说、只剩静止画面）两种偏差，方向还相反。
        # 旁白总字数直接决定成片有没有长时间空档，所以按实际字数兜一次改写。
        # 只多打一次文本模型，比让整条视频作废便宜得多。
        min_total = max(shot_count * per_shot_floor, int(total_chars * 0.85))
        total = self._narration_total(shots)
        attempts = 0
        # 实测一轮改写不一定能落进区间（29 字改成 42 字，仍偏低），所以给两轮。
        # 文本模型一次调用远比让整条视频带着空档或断句交付便宜。
        while attempts < NARRATION_REWRITE_ATTEMPTS and not min_total <= total <= total_chars:
            messages.append(
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"你上一版每镜旁白分别是 "
                        f"{'、'.join(str(len(str(s.get('narration', '')))) for s in shots)} 字，"
                        f"全片 {total} 字。"
                        f"请把每镜 narration 改写成正好 {per_shot_chars} 字左右"
                        f"（全片约 {total_chars} 字，不得少于 {min_total} 字，不得多于 {total_chars} 字）。"
                        "保留同样的分镜结构、主体锁定、画面描述和事实依据；"
                        "每镜多写一个具体的景物、动作或可核实的细节来补足字数，"
                        "不要用重复句、空话或感叹词凑数。"
                        "subtitle 与改写后的 narration 逐字对应。重新输出完整 JSON。"
                    ),
                }
            )
            result, shots = self._generate_validated(messages, shot_count)
            total = self._narration_total(shots)
            attempts += 1

        # AI 腔词表扫描：命中触发一轮整体改写（见 COPY_REWRITE_ATTEMPTS 注释）。
        # 改写后仍命中不再追加轮次，剩余命中如实写进 event，供评测与人工复核。
        copy_scan = scan_script(result)
        copy_attempts = 0
        while copy_attempts < COPY_REWRITE_ATTEMPTS and not copy_scan["ok"]:
            messages.append(
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": self._copy_rewrite_instruction(
                        copy_scan["hits"],
                        per_shot_floor=per_shot_floor,
                        per_shot_chars=per_shot_chars,
                        total_chars=total_chars,
                    ),
                }
            )
            result, shots = self._generate_validated(messages, shot_count)
            copy_scan = scan_script(result)
            copy_attempts += 1

        return LLMResult(
            data=result,
            provider=self.name,
            model=self.model,
            fallback_used=False,
            event={
                "type": "llm",
                "provider": self.name,
                "model": self.model,
                "ok": True,
                "narration_chars": self._narration_total(shots),
                "narration_target": total_chars,
                "narration_rewrite_attempts": attempts,
                "billing_instrumented": True,
                "prompt_version": "zj-tourism-prompt-v3",
                "subject_lock": subject_lock,
                "copy_scan": {
                    "ok": copy_scan["ok"],
                    "hit_count": len(copy_scan["hits"]),
                    "prompt_hit_count": len(copy_scan["prompt_hits"]),
                    "terms": hit_terms(copy_scan["hits"]),
                    "rewrite_attempts": copy_attempts,
                },
            },
        )

    @staticmethod
    def _build_instruction(
        prompt: str,
        context: dict[str, Any],
        *,
        shot_count: int,
        duration: int,
        budget: dict[str, float],
        subject_lock: str,
        knowledge_context: str,
        custom_brief: str,
    ) -> str:
        """拼装脚本生成指令；独立成函数便于单测与评测脚本复用同一口径。"""
        per_shot_seconds = budget["per_shot_seconds"]
        per_shot_chars = budget["per_shot_chars"]
        per_shot_floor = budget["per_shot_floor"]
        total_chars = budget["total_chars"]
        style = str(context.get("style", "")).strip()
        style_voice = STYLE_VOICES.get(style, STYLE_VOICE_DEFAULT)
        schema = (
            '{"title":"string","summary":"string","cta":"string",'
            '"tags":["string"],"narration":"string",'
            '"shots":[{"id":"shot_1","order":1,"duration_sec":5,'
            '"title":"string","visual_prompt":"string","narration":"string",'
            '"subtitle":"string","transition":"fade"}],'
            '"knowledge":{"fact":"string","color":"#24526a","accent":"#d6ad60","aliases":[]}}'
        )
        fallback_rule = (
            "不得编造实时价格、活动日期、交通班次、荣誉或其他无法核验的事实。"
            if not knowledge_context
            else "实时价格、活动日期、交通班次、荣誉等资料未覆盖的信息不得编造。"
        )
        return (
            "你是浙江文旅竖屏短视频总编导。请根据用户输入生成可直接用于视频制作的 JSON。"
            f"本次叙事风格是「{style or '未指定'}」，旁白语气要求：{style_voice}\n"
            f"{COPY_STYLE_RULES}"
            "旁白取材硬约束：每个镜头的 narration 与 visual_prompt 都必须用到知识库条目里的"
            "「画面提示」短语或「可用事实」中的具体名词（名称、数字、材质、年份）至少一处；"
            "资料块为空时，改为从用户输入的城市、地标、主题里取具体细节。\n"
            f"必须生成 {shot_count} 个镜头，镜头时长总和必须等于 {duration} 秒。"
            f"每镜约 {per_shot_seconds:.1f} 秒，中文配音约每秒 {NARRATION_CHARS_PER_SECOND} 字，"
            f"该镜的 narration 应写满 {per_shot_floor}~{per_shot_chars} 个汉字——"
            f"这是目标区间，不是上限。"
            f"全片旁白合计不得超过 {total_chars} 个汉字，这是硬约束："
            f"配音时长 = 总字数 ÷ {NARRATION_CHARS_PER_SECOND} + 镜头间的停顿，超过用户选的 {duration} 秒，"
            "最后一句的结尾就会被裁掉。镜头多或少都要先把总字数守住。"
            "旁白太短会在成片里留下没有解说、只有静止画面的空档，太长则结尾被裁掉。"
            "按目标字数写满，不要靠重复句或空话凑数。"
            "subtitle 是烧进画面给观众看的字幕，必须与同一镜 narration 逐字对应"
            f"（可用空格代替逗号、去掉句末标点），长度不超过 {per_shot_chars} 字；"
            "不要另起一个短标题，观众读到的字必须和听到的话一致。"
            "每个镜头必须有具体可视化画面、旁白和字幕，不能使用空字符串、占位符或泛泛的“风景”。"
            f"全片必须锁定同一主体：{subject_lock}。"
            "每个 visual_prompt 都要明确写出这个主体如何出现，并只改变机位、景别、动作、时间和光线；"
            "不得在不同镜头重新设计主体，也不要用风格形容词代替主体描述。"
            "visual_prompt 的推荐结构是：主体及其固定外观 + 本镜场景/动作 + 景别与机位 + 光线；"
            "不要写抽象空话，不要指定文字、水印或画面内字幕。"
            f"{fallback_rule}"
            f"{knowledge_context}"
            + FEWSHOT_EXAMPLES
            + "只输出 JSON，不要 Markdown，不要解释。JSON 结构必须符合："
            f"{schema}"
            f"用户补充要求：{custom_brief or '无'}。"
            f"用户输入：{prompt}"
        )

    @staticmethod
    def _copy_rewrite_instruction(
        hits: list[dict[str, Any]],
        *,
        per_shot_floor: int,
        per_shot_chars: int,
        total_chars: int,
    ) -> str:
        terms = "、".join(hit_terms(hits))
        fields = format_hit_fields(hits)
        return (
            f"你上一版文案命中了 AI 腔词表：{terms}（{fields}）。"
            "请整体改写 narration 与 subtitle："
            "1) 全部去掉命中的词与句式，也不要换成同义套话；"
            "2) 抽象形容词换成具体名词——景物、动作、声音、材质、数字；"
            f"3) 每镜字数保持在 {per_shot_floor}~{per_shot_chars} 字，全片不超过 {total_chars} 字；"
            "4) subtitle 与改写后的 narration 逐字对应；"
            "5) 保留分镜结构、主体锁定、visual_prompt 与事实依据。重新输出完整 JSON。"
        )

    # 五要素校验的打回轮数。与词表改写同一哲学：一轮收敛不了就如实报告，
    # 不为完美指标无限制烧调用。
    REFINE_ELEMENT_REWRITE_ATTEMPTS = 1

    def refine_prompts(
        self, shots: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        """视觉 prompt 精修：把脚本阶段顺手写的画面描述升级为五要素生成 prompt。

        返回 {"refined": [...], "event": {...}}，refined 与输入按 id 一一对应。
        解析失败抛错由调用方降级（保留原 prompt 继续跑）；五要素两轮仍不齐时
        不抛错，把 pass_rate 如实写进 event——精修是增强，不能反向阻断主流程。
        """
        if not shots:
            return {"refined": [], "event": {"type": "prompt_refine", "ok": True, "refined": 0}}
        subject_lock = str(context.get("subject_lock", "")).strip()
        if not subject_lock:
            place = f"{context.get('city', '')}{context.get('landmark', '')}".strip()
            subject_lock = f"{place or '核心景点'}的可辨识主体、轮廓、材质和标志性细节"
        style = str(context.get("style", "")).strip()
        messages = [
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": _refine_instruction(shots, style, subject_lock)},
        ]
        result = self._chat(messages)
        refined = self._validated_refined(result, shots)
        missing_map = self._missing_elements(refined)
        attempts = 0
        while attempts < self.REFINE_ELEMENT_REWRITE_ATTEMPTS and any(missing_map.values()):
            report = "；".join(
                f"{shot_id} 缺：{'、'.join(missing)}"
                for shot_id, missing in missing_map.items()
                if missing
            )
            messages.append(
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"以下镜头的 visual_prompt 五要素不完整：{report}。"
                        "请补全缺失要素后重新输出完整 JSON；已合格的镜头原样保留。"
                    ),
                }
            )
            result = self._chat(messages)
            refined = self._validated_refined(result, shots)
            missing_map = self._missing_elements(refined)
            attempts += 1
        total = len(refined)
        passed = sum(1 for missing in missing_map.values() if not missing)
        event = {
            "type": "prompt_refine",
            "ok": passed == total,
            "refined": total,
            "five_element_pass": passed,
            "five_element_total": total,
            "five_element_pass_rate": round(passed / total, 4) if total else 0.0,
            "element_rewrite_attempts": attempts,
        }
        return {"refined": refined, "event": event}

    @staticmethod
    def _missing_elements(refined: list[dict[str, Any]]) -> dict[str, list[str]]:
        return {
            str(item.get("id", "")): missing_visual_prompt_elements(item.get("visual_prompt", ""))
            for item in refined
        }

    @staticmethod
    def _validated_refined(
        result: dict[str, Any], shots: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """校验精修返回：逐 id 对齐、visual_prompt 非空、search_queries 最多 3 条。"""
        items = result.get("shots")
        if not isinstance(items, list) or len(items) != len(shots):
            count = len(items) if isinstance(items, list) else 0
            raise RuntimeError(f"refine response returned {count} shots; expected {len(shots)}")
        expected_ids = [str(shot.get("id")) for shot in shots]
        by_id: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError("refine response has an invalid entry")
            key = str(item.get("id", ""))
            if key not in expected_ids:
                raise RuntimeError(f"refine response returned unknown shot id: {key}")
            if not str(item.get("visual_prompt", "")).strip():
                raise RuntimeError(f"refine response has an empty visual_prompt for {key}")
            by_id[key] = item
        if len(by_id) != len(expected_ids):
            raise RuntimeError("refine response is missing some shots")
        return [
            {
                "id": shot_id,
                "visual_prompt": str(by_id[shot_id].get("visual_prompt", "")).strip(),
                "search_queries": [
                    str(query).strip()
                    for query in (by_id[shot_id].get("search_queries") or [])
                    if str(query).strip()
                ][:3],
            }
            for shot_id in expected_ids
        ]

    # —— 候选池与逐镜详述（explore 模式）——
    def candidates(self, context: dict[str, Any]) -> LLMResult:
        """生成 10 个候选方向；校验不过时带问题清单重写，最多 2 轮。"""
        knowledge_context = _resolve_knowledge_context(context)

        def _field(name: str) -> str:
            value = context.get(name)
            return "" if value is None else str(value)

        target = candidate_service.target_shot_count(
            int(context.get("duration") or 30), int(context.get("shot_count") or 0)
        )
        total, roles = candidate_service.candidate_plan(target)
        instruction = candidate_service.build_candidates_instruction(
            city=_field("city"),
            landmark=_field("landmark"),
            theme=_field("theme"),
            culture=_field("culture"),
            festival=_field("festival"),
            audience=_field("audience"),
            style=_field("style"),
            custom_brief=_field("custom_brief"),
            knowledge_context=knowledge_context,
            total=total,
            roles=roles,
        )
        messages = [
            {"role": "system", "content": "你是严谨的浙江文旅分镜策划，只输出合法 JSON。"},
            {"role": "user", "content": instruction},
        ]
        payload = self._chat(messages)
        arc, candidates = candidate_service.normalize_candidates(payload)
        problems = candidate_service.validate_candidates(
            candidates, city=_field("city"), expected_total=total, roles=roles
        )
        attempts = 0
        while problems and attempts < candidate_service.CANDIDATE_REWRITE_ATTEMPTS:
            messages.append(
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)[:6000]}
            )
            messages.append(
                {"role": "user", "content": candidate_service.candidate_rewrite_instruction(problems)}
            )
            payload = self._chat(messages)
            arc, candidates = candidate_service.normalize_candidates(payload)
            problems = candidate_service.validate_candidates(
                candidates, city=_field("city"), expected_total=total, roles=roles
            )
            attempts += 1
        if problems:
            raise RuntimeError("候选池校验未通过：" + "；".join(problems[:6]))
        return LLMResult(
            data={
                "arc": arc,
                "candidates": [item.to_dict() for item in candidates],
                "target_shots": target,
                "candidate_total": total,
            },
            provider=self.name,
            model=self.model,
            fallback_used=False,
            event={
                "type": "shot_candidates",
                "provider": self.name,
                "model": self.model,
                "count": len(candidates),
                "target_shots": target,
                "rewrite_attempts": attempts,
                "prompt_version": "candidates-v1",
            },
        )

    def detail(self, context: dict[str, Any]) -> dict[str, Any]:
        """为已选镜头生成结构化详述；校验不过时只回炉出问题的镜头。"""
        selected = list(context.get("selected") or [])
        per_shot_seconds = [float(value) for value in (context.get("per_shot_seconds") or [])]
        task_input = dict(context.get("task_input") or {})
        style_ctx = dict(context.get("style_ctx") or {})
        if not selected or len(selected) != len(per_shot_seconds):
            raise RuntimeError("详述上下文缺少选择的镜头或时长")
        budget = plan_narration_budget(int(task_input.get("duration", 30)), len(selected))
        knowledge_context = _resolve_knowledge_context(task_input)
        messages = [
            {"role": "system", "content": "你是严谨的浙江文旅分镜摄影师，只输出合法 JSON。"},
            {
                "role": "user",
                "content": detail_service.build_detail_instruction(
                    selected=selected,
                    task_input=task_input,
                    knowledge_context=knowledge_context,
                    narration_budget=budget,
                    per_shot_seconds=per_shot_seconds,
                ),
            },
        ]
        payload = self._chat(messages)
        shots = detail_service.normalize_detail(
            payload, selected=selected, per_shot_seconds=per_shot_seconds
        )
        gate = detail_service.validate_detail(
            shots,
            selected=selected,
            theme=str(task_input.get("theme") or ""),
            city=str(task_input.get("city") or ""),
            narration_budget=budget,
            composed=detail_service.build_composed_prompts(shots, style_ctx),
        )
        attempts = 0
        while gate.problems and attempts < detail_service.DETAIL_REWRITE_ATTEMPTS:
            failing_ids = set(detail_service.failing_shot_ids(gate))
            failing = [shot for shot in shots if shot["id"] in failing_ids]
            if not failing:
                break
            messages = [
                {"role": "system", "content": "你是严谨的浙江文旅分镜摄影师，只输出合法 JSON。"},
                {
                    "role": "user",
                    "content": detail_service.build_selective_rewrite_instruction(
                        failing=failing, per_shot=gate.per_shot
                    ),
                },
            ]
            patch = self._chat(messages)
            shots = detail_service.merge_rewritten(
                shots, patch, selected=selected, per_shot_seconds=per_shot_seconds
            )
            gate = detail_service.validate_detail(
                shots,
                selected=selected,
                theme=str(task_input.get("theme") or ""),
                city=str(task_input.get("city") or ""),
                narration_budget=budget,
                composed=detail_service.build_composed_prompts(shots, style_ctx),
            )
            attempts += 1
        return {
            "title": str(payload.get("title") or ""),
            "summary": str(payload.get("summary") or ""),
            "cta": str(payload.get("cta") or ""),
            "shots": shots,
            "gate": {
                "ok": gate.ok,
                "problems": gate.problems,
                "composed_lengths": gate.composed_lengths,
                "rewrite_attempts": attempts,
            },
            "event": {
                "type": "shot_detail",
                "provider": self.name,
                "model": self.model,
                "shot_count": len(shots),
                "gate_ok": gate.ok,
                "problem_count": len(gate.problems),
                "rewrite_attempts": attempts,
                "prompt_version": "detail-v1",
            },
        }

    def _generate_validated(
        self, messages: list[dict[str, str]], shot_count: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Parse JSON, then validate the script; repair once on either failure."""
        result = self._chat(messages)
        try:
            return result, self._validated_shots(result, shot_count)
        except Exception as exc:  # noqa: BLE001 - retry once with the exact validation error
            repair_messages = list(messages) + [
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)[:6000]},
                {
                    "role": "user",
                    "content": (
                        f"上一版不符合要求：{exc}。请修正后重新输出完整 JSON；"
                        "镜头数、必填字段、字幕与旁白逐字对应、转场名都必须正确。"
                        "只输出 JSON，不要 Markdown 或解释。"
                    ),
                },
            ]
            result = self._chat(repair_messages)
            return result, self._validated_shots(result, shot_count)

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        content = self._complete(messages)
        try:
            return _parse_json_object(content)
        except (json.JSONDecodeError, ValueError) as exc:
            repair_messages = list(messages) + [
                {"role": "assistant", "content": str(content or "")[:6000]},
                {
                    "role": "user",
                    "content": (
                        f"上一条回复不是合法 JSON：{exc}。"
                        "请只输出修正后的完整 JSON，不要 Markdown 代码块或解释。"
                    ),
                },
            ]
            content = self._complete(repair_messages)
            try:
                return _parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as second:
                raise RuntimeError(f"LLM returned invalid JSON after repair: {second}") from second

    def _complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "temperature": 0.7,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with provider_call("llm", self.name, self.model) as billing:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                record_response(billing, response)
                payload = json.loads(response.read().decode("utf-8"))
                record_response(billing, response, payload)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response is missing choices[0].message.content") from exc
        return str(content or "")

    @staticmethod
    def _narration_total(shots: list[dict[str, Any]]) -> int:
        return sum(len(str(shot.get("narration", ""))) for shot in shots)

    @staticmethod
    def _validated_shots(result: dict[str, Any], shot_count: int) -> list[dict[str, Any]]:
        shots = result.get("shots")
        if not isinstance(shots, list) or len(shots) != shot_count:
            count = len(shots) if isinstance(shots, list) else 0
            raise RuntimeError(f"LLM returned {count} shots; expected {shot_count}")
        required = ("title", "summary", "cta", "narration")
        if any(not str(result.get(key, "")).strip() for key in required):
            raise RuntimeError("LLM returned an empty title, summary, CTA, or narration")
        for index, shot in enumerate(shots, start=1):
            if not isinstance(shot, dict):
                raise RuntimeError("LLM returned an invalid shot")
            for key in ("title", "visual_prompt", "narration", "subtitle"):
                if not str(shot.get(key, "")).strip():
                    raise RuntimeError(f"Shot {index} returned an empty {key}")
            if _caption_key(shot.get("subtitle")) != _caption_key(shot.get("narration")):
                raise RuntimeError(f"Shot {index} subtitle does not match its narration")
            if "duration_sec" in shot:
                try:
                    duration = float(shot["duration_sec"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Shot {index} has a non-numeric duration_sec") from exc
                if not math.isfinite(duration) or duration <= 0:
                    raise RuntimeError(f"Shot {index} has an invalid duration_sec")
            transition = str(shot.get("transition") or "").strip().lower()
            if transition and transition not in TRANSITIONS:
                raise RuntimeError(f"Shot {index} has an unsupported transition: {transition}")
            shot.setdefault("id", f"shot_{index}")
            shot.setdefault("order", index)
            shot["transition"] = transition or ("fade" if index == 1 else "dissolve")
        return shots
