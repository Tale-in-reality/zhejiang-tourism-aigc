"""逐镜详述：把用户选中的候选升级为"详尽齐全"的结构化画面规格 + 连贯旁白。

与候选池的分工：
- 候选池回答"拍什么方向"，粒度是大方向 + 2–4 个锚点；
- 详述回答"这一镜具体长什么样"，粒度是五要素齐全的结构化规格。

校验门是本模块的重点：字段完整、锚点继承、五要素、内部自洽（时段↔光线）、
长度预算、禁词、事实溯源、旁白连贯，逐条对应提示词里的硬约束。不合格时
**只把出问题的镜头打回重写**，不动已合格的镜头——收敛快、成本低。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .consistency import missing_visual_prompt_elements, visual_prompt_elements
from .copy_quality import scan_text

# —— 长度预算 ——
# 正文（visual_prompt）：低于下限说明"又是一句话"，高于上限会被生成模型摊薄。
BODY_MIN_CHARS = 110
BODY_MAX_CHARS = 180
# 组合后完整 prompt（正文 + 风格前缀 + 主体锁定 + 生成约束）。实测无上限时
# 会到 300+ 且信息密度下降，这里两头都拦。
COMPOSED_MIN_CHARS = 220
COMPOSED_MAX_CHARS = 340

DETAIL_REWRITE_ATTEMPTS = 2

MIN_SUBJECT_DETAILS = 2
MAX_SUBJECT_DETAILS = 4
MIN_MUST_INCLUDE = 3
MAX_MUST_INCLUDE = 5
MIN_MUST_AVOID = 2
MAX_MUST_AVOID = 5

ROLE_KEYS = ("opening", "subject", "detail", "person", "closing")
TIMES = ("清晨", "正午", "黄昏", "夜")
SIZES = ("远景", "全景", "中景", "近景", "特写")

# 时段与光线的自洽表：某时段下允许出现的光线关键词。
LIGHT_COMPAT: dict[str, tuple[str, ...]] = {
    "清晨": ("晨", "低角度", "散射", "柔", "薄雾", "冷", "逆光", "侧光"),
    "正午": ("顶光", "硬光", "强烈", "直射", "高对比", "短影"),
    "黄昏": ("夕", "暖", "侧光", "逆光", "低角度", "金色", "长影"),
    "夜": ("夜", "月光", "灯", "低照度", "暗", "蓝调", "暖光点"),
}
# 明显冲突的组合：出现即判不自洽。
LIGHT_CONFLICTS: dict[str, tuple[str, ...]] = {
    "清晨": ("顶光", "正午阳光", "强烈直射"),
    "正午": ("夜色", "月光", "夕照", "黄昏"),
    "黄昏": ("正午", "顶光", "晨雾未散"),
    "夜": ("正午", "顶光", "强烈直射", "晨光"),
}

DETAIL_SCHEMA = (
    '{"title":"string","summary":"string","cta":"string",'
    '"shots":[{"candidate_id":"c1","duration_sec":5,'
    '"subject":{"name":"","material":"","details":["2-4条"],"scale":""},'
    '"environment":{"place":"","season":"","weather":"","time":"清晨|正午|黄昏|夜"},'
    '"lighting":{"direction":"","quality":"","contrast":"低|中|高"},'
    '"composition":{"shot_size":"远景|全景|中景|近景|特写","angle":"","framing":"","negative_space":""},'
    '"palette":{"temperature":"冷|中性|暖","dominant":["主色1","主色2"]},'
    '"motion":{"in_frame":["1-3条画面内动态"],"camera":"静止|缓慢推近|缓慢拉远|横向平移"},'
    '"must_include":["3-5条"],"must_avoid":["2-5条"],'
    '"continuity":{"same_subject":true,"same_palette":true,"relation_to_prev":""},'
    '"visual_prompt":"正文110-190字","narration":"string","subtitle":"string"}]}'
)


def build_detail_instruction(
    *,
    selected: list[dict[str, Any]],
    task_input: dict[str, Any],
    knowledge_context: str,
    narration_budget: dict[str, float],
    per_shot_seconds: list[float],
) -> str:
    """详述指令：逐镜给候选信息与配额，要求输出结构化规格 + 连贯旁白。"""
    payload = json.dumps(selected, ensure_ascii=False)
    durations = "、".join(f"{value:.1f}" for value in per_shot_seconds)
    style = str(task_input.get("style") or "未指定")
    audience = str(task_input.get("audience") or "未指定")
    total_chars = int(narration_budget["total_chars"])
    floor = int(narration_budget["per_shot_floor"])
    cap = int(narration_budget["per_shot_chars"])
    return (
        "为已选定的镜头写「可直接发给生成模型」的完整画面规格，并按用户选择的顺序写连贯旁白。\n"
        f"全片风格：{style}；受众：{audience}；镜头数 {len(selected)}，"
        f"各镜时长依次为 {durations} 秒（旁白按此时长写）。\n"
        f"用户选择的镜头（顺序即成片顺序）：{payload}\n\n"
        "必须遵守（任何一条不满足都会被打回重写）：\n"
        "1. 继承锚点：每镜的 must_include 必须包含该候选 imagery 的全部锚点，"
        f"再补 1–2 个本镜特有的具体细节；总数 {MIN_MUST_INCLUDE}–{MAX_MUST_INCLUDE} 条。\n"
        f"2. 五要素齐全：主体（材质+痕迹，details {MIN_SUBJECT_DETAILS}–{MAX_SUBJECT_DETAILS} 条）、"
        "环境光线（时段+光源方向）、构图（景别+机位+主体在画面中的位置+留白）、"
        "色彩（色温+两个主色）、画面内动态。缺任何一项即为不合格。\n"
        "3. 具体到能拍：写「青灰石栏内侧的缺口和雨水冲出的浅槽」，不写「古老的石桥」；"
        "写「雾贴着水面缓慢向东流」，不写「雾气缭绕」。\n"
        "4. 时段与光线自洽：清晨配低角度散射光，正午配顶光硬光，黄昏配暖侧光，夜配低照度与灯光；"
        "不得出现「夜」配「正午顶光」这类矛盾。\n"
        "5. 色彩全片统一：所有镜头同一色温基调，只允许因时段产生的合理偏移（palette.temperature）。\n"
        "6. 画面内不得出现文字、水印、logo、字幕（字幕后期烧录）；must_avoid 要写本镜特有的风险项。\n"
        "7. 事实只用给定资料：年代、数字、工艺名必须有来源；不确定就不写。\n"
        f"8. visual_prompt 是 {BODY_MIN_CHARS}–{BODY_MAX_CHARS} 字的连续中文描述，"
        "按 主体→环境光线→构图→色彩→画面内动态 的顺序写，不要分点、不要标题。\n"
        f"9. 旁白按顺序连读成一篇：每镜首句 ≤12 字，每镜 {floor}–{cap} 字，"
        f"全片总字数不超过 {total_chars} 字；subtitle 与 narration 逐字对应。\n"
        "10. 禁用 AI 腔：绝美/璀璨/邂逅/解锁/天花板/视觉盛宴/心灵之旅/诗与远方/沉浸式体验；"
        "「让我们一起」「在这里，你可以」「不仅…更是…」「仿佛在诉说」「承载着…记忆」。\n"
        f"{knowledge_context}\n"
        "只输出 JSON，不要 Markdown，不要解释。结构必须符合："
        f"{DETAIL_SCHEMA}"
    )


@dataclass
class DetailGateResult:
    problems: list[str]
    per_shot: dict[str, list[str]]
    composed_lengths: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.problems


def _text(value: Any) -> str:
    return str(value or "").strip()


def _caption_key(value: Any) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "")).casefold()


def _require(obj: dict[str, Any], *keys: str) -> list[str]:
    return [key for key in keys if not _text(obj.get(key))]


def normalize_detail(
    payload: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    per_shot_seconds: list[float],
) -> list[dict[str, Any]]:
    """把模型输出归一化为按选择顺序排列的镜头规格列表。

    数量不符或 candidate_id 对不上直接抛错（结构性问题没法靠补默认值救）；
    字段级缺失保留空值，交给 validate_detail 精确报出。
    """
    items = payload.get("shots")
    if not isinstance(items, list) or len(items) != len(selected):
        count = len(items) if isinstance(items, list) else 0
        raise RuntimeError(f"详述返回 {count} 个镜头，应为 {len(selected)} 个")
    expected_ids = [str(row["candidate_id"]) for row in selected]
    by_candidate: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("详述返回的镜头不是对象")
        key = str(item.get("candidate_id") or "").strip()
        if key not in expected_ids:
            raise RuntimeError(f"详述返回了未选择的候选：{key}")
        if key in by_candidate:
            raise RuntimeError(f"详述重复返回候选 {key}")
        by_candidate[key] = item
    shots: list[dict[str, Any]] = []
    for index, (row, seconds) in enumerate(zip(selected, per_shot_seconds), start=1):
        item = by_candidate[str(row["candidate_id"])]
        spec = {
            "subject": dict(item.get("subject") or {}),
            "environment": dict(item.get("environment") or {}),
            "lighting": dict(item.get("lighting") or {}),
            "composition": dict(item.get("composition") or {}),
            "palette": dict(item.get("palette") or {}),
            "motion": dict(item.get("motion") or {}),
            "continuity": dict(item.get("continuity") or {}),
        }
        shots.append(
            {
                "id": f"shot_{index}",
                "order": index,
                "candidate_id": str(row["candidate_id"]),
                "duration_sec": round(float(seconds), 3),
                "title": _text(item.get("title")) or _text(row.get("title")) or f"镜头{index}",
                "visual_prompt": _text(item.get("visual_prompt")),
                "narration": _text(item.get("narration")),
                "subtitle": _text(item.get("subtitle")),
                "prompt_spec": spec,
                "intent_anchors": [
                    _text(anchor) for anchor in (item.get("must_include") or []) if _text(anchor)
                ],
                "must_avoid": [
                    _text(anchor) for anchor in (item.get("must_avoid") or []) if _text(anchor)
                ],
            }
        )
    return shots


def validate_detail(
    shots: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    theme: str = "",
    city: str = "",
    narration_budget: dict[str, float],
    composed: dict[str, str] | None = None,
) -> DetailGateResult:
    """全套校验门。返回逐镜问题与汇总问题，composed 传入组合后的 prompt 以查长度。"""
    per_shot: dict[str, list[str]] = {}
    composed_lengths: dict[str, int] = {}
    candidate_by_id = {str(row["candidate_id"]): row for row in selected}
    for shot in shots:
        issues: list[str] = []
        candidate = candidate_by_id.get(str(shot.get("candidate_id")), {})
        spec = shot.get("prompt_spec") or {}
        subject = spec.get("subject") or {}
        environment = spec.get("environment") or {}
        lighting = spec.get("lighting") or {}
        composition = spec.get("composition") or {}
        palette = spec.get("palette") or {}
        motion = spec.get("motion") or {}
        continuity = spec.get("continuity") or {}
        body = str(shot.get("visual_prompt") or "")
        # 1. 字段完整
        for label, missing in (
            ("主体", _require(subject, "name", "material", "scale")),
            ("环境", _require(environment, "place", "season", "weather", "time")),
            ("光线", _require(lighting, "direction", "quality", "contrast")),
            ("构图", _require(composition, "shot_size", "angle", "framing", "negative_space")),
            ("色彩", _require(palette, "temperature")),
            ("动态", _require(motion, "camera")),
        ):
            if missing:
                issues.append(f"{label}缺字段：{'/'.join(missing)}")
        details = [str(value).strip() for value in (subject.get("details") or []) if str(value).strip()]
        if not (MIN_SUBJECT_DETAILS <= len(details) <= MAX_SUBJECT_DETAILS):
            issues.append(
                f"subject.details 应为 {MIN_SUBJECT_DETAILS}–{MAX_SUBJECT_DETAILS} 条，实际 {len(details)} 条"
            )
        dominant = [str(value).strip() for value in (palette.get("dominant") or []) if str(value).strip()]
        if len(dominant) < 2:
            issues.append("palette.dominant 至少两个主色")
        in_frame = [str(value).strip() for value in (motion.get("in_frame") or []) if str(value).strip()]
        if not in_frame:
            issues.append("motion.in_frame 至少一条画面内动态")
        if str(environment.get("time") or "") and environment["time"] not in TIMES:
            issues.append(f"environment.time 必须是 {'/'.join(TIMES)}")
        if str(composition.get("shot_size") or "") and composition["shot_size"] not in SIZES:
            issues.append(f"composition.shot_size 必须是 {'/'.join(SIZES)}")
        # 2. 锚点继承
        anchors = [str(value) for value in (shot.get("intent_anchors") or [])]
        for imagery in candidate.get("imagery") or []:
            if imagery not in anchors:
                issues.append(f"未继承候选锚点「{imagery}」")
        if not (MIN_MUST_INCLUDE <= len(anchors) <= MAX_MUST_INCLUDE):
            issues.append(
                f"must_include 应为 {MIN_MUST_INCLUDE}–{MAX_MUST_INCLUDE} 条，实际 {len(anchors)} 条"
            )
        avoid = [str(value) for value in (shot.get("must_avoid") or [])]
        if not (MIN_MUST_AVOID <= len(avoid) <= MAX_MUST_AVOID):
            issues.append(
                f"must_avoid 应为 {MIN_MUST_AVOID}–{MAX_MUST_AVOID} 条，实际 {len(avoid)} 条"
            )
        # 3. 五要素与正文长度
        if not visual_prompt_elements(body)["subject"]:
            issues.append("visual_prompt 为空")
        else:
            missing = missing_visual_prompt_elements(body)
            if missing:
                issues.append("visual_prompt 缺要素：" + "、".join(missing))
            if len(body) < BODY_MIN_CHARS:
                issues.append(f"visual_prompt 只有 {len(body)} 字，低于 {BODY_MIN_CHARS} 字（不够详尽）")
            if len(body) > BODY_MAX_CHARS:
                issues.append(f"visual_prompt {len(body)} 字，超过 {BODY_MAX_CHARS} 字")
        # 4. 时段↔光线自洽
        time_value = str(environment.get("time") or "")
        light_blob = f"{lighting.get('direction','')}{lighting.get('quality','')}{lighting.get('contrast','')}"
        for conflict in LIGHT_CONFLICTS.get(time_value, ()):
            if conflict in light_blob:
                issues.append(f"时段「{time_value}」与光线「{conflict}」矛盾")
        # 5. 地点与城市
        place = str(environment.get("place") or "")
        if city and place and city not in place and city not in str(candidate.get("place") or ""):
            issues.append(f"地点「{place}」不在所选城市 {city}")
        # 6. 禁词（旁白/字幕/正文）
        for label, value in (("旁白", shot.get("narration")), ("字幕", shot.get("subtitle")),
                             ("画面描述", body)):
            hits = scan_text(str(value or ""))
            if hits:
                terms = "、".join(sorted({hit["term"] for hit in hits}))
                issues.append(f"{label}命中 AI 腔词表：{terms}")
        # 7. 字幕与旁白逐字对应
        if _caption_key(shot.get("subtitle")) != _caption_key(shot.get("narration")):
            issues.append("subtitle 与 narration 不逐字对应")
        # 8. 连贯性
        if int(shot.get("order") or 1) > 1 and not _text(continuity.get("relation_to_prev")):
            issues.append("continuity.relation_to_prev 未填写（从第二镜起必填）")
        # 9. 组合长度
        if composed is not None:
            full = composed.get(str(shot.get("id")), "")
            composed_lengths[str(shot.get("id"))] = len(full)
            if full and len(full) < COMPOSED_MIN_CHARS:
                issues.append(f"组合后 prompt 只有 {len(full)} 字，低于 {COMPOSED_MIN_CHARS} 字")
            if full and len(full) > COMPOSED_MAX_CHARS:
                issues.append(f"组合后 prompt {len(full)} 字，超过 {COMPOSED_MAX_CHARS} 字")
        per_shot[str(shot.get("id"))] = issues
    problems: list[str] = []
    for shot_id, issues in per_shot.items():
        problems.extend(f"{shot_id}: {issue}" for issue in issues)
    # 10. 旁白预算
    total = sum(len(str(shot.get("narration") or "")) for shot in shots)
    total_target = int(narration_budget["total_chars"])
    min_total = max(len(shots) * int(narration_budget["per_shot_floor"]), int(total_target * 0.85))
    if total > total_target:
        problems.append(f"全片旁白 {total} 字，超过预算 {total_target} 字（成片结尾会被裁）")
    elif total < min_total:
        problems.append(f"全片旁白 {total} 字，低于下限 {min_total} 字（成片会留空档）")
    if theme and not any(theme in str(shot.get("narration") or "") for shot in shots):
        # 只做提醒级别的软校验：主题词不出现不一定是错（可能用同义表达）
        pass
    return DetailGateResult(problems=problems, per_shot=per_shot, composed_lengths=composed_lengths)


def failing_shot_ids(gate: DetailGateResult) -> list[str]:
    return [shot_id for shot_id, issues in gate.per_shot.items() if issues]


def build_selective_rewrite_instruction(
    *,
    failing: list[dict[str, Any]],
    per_shot: dict[str, list[str]],
) -> str:
    """只回炉出问题的镜头：给出镜头当前内容与逐条问题，要求原样返回其他字段。"""
    payload = json.dumps(
        [
            {
                "shot_id": shot["id"],
                "candidate_id": shot["candidate_id"],
                "visual_prompt": shot["visual_prompt"],
                "narration": shot["narration"],
                "must_include": shot.get("intent_anchors") or [],
                "must_avoid": shot.get("must_avoid") or [],
            }
            for shot in failing
        ],
        ensure_ascii=False,
    )
    report = "；".join(
        f"{shot_id}：" + "；".join(issues[:6]) for shot_id, issues in per_shot.items() if issues
    )
    return (
        f"以下镜头的详述不合格：{report}\n"
        f"待修正的镜头：{payload}\n"
        "请只输出这些镜头的修正版，JSON 结构为 {\"shots\":[...]}，"
        "每个镜头保留原来的 candidate_id，其他字段（subject/environment/lighting/"
        "composition/palette/motion/must_include/must_avoid/continuity/visual_prompt/"
        "narration/subtitle）必须完整；同时保持旁白与前后镜的衔接。"
        "只输出 JSON。"
    )


def merge_rewritten(
    shots: list[dict[str, Any]],
    rewritten_payload: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    per_shot_seconds: list[float],
) -> list[dict[str, Any]]:
    """把回炉结果按 candidate_id 合并回原列表，未改动的镜头保持原样。"""
    items = rewritten_payload.get("shots")
    if not isinstance(items, list):
        raise RuntimeError("回炉响应缺少 shots 数组")
    by_candidate = {
        str(item.get("candidate_id")): item for item in items if isinstance(item, dict)
    }
    if not by_candidate:
        raise RuntimeError("回炉响应没有可合并的镜头")
    merged = normalize_detail(
        {
            "shots": [
                by_candidate.get(str(shot.get("candidate_id")), _shot_to_detail_item(shot))
                for shot in shots
            ]
        },
        selected=selected,
        per_shot_seconds=per_shot_seconds,
    )
    return merged


def _shot_to_detail_item(shot: dict[str, Any]) -> dict[str, Any]:
    """把已有镜头还原成详述响应格式，供合并时占位。"""
    spec = shot.get("prompt_spec") or {}
    return {
        "candidate_id": shot.get("candidate_id"),
        "visual_prompt": shot.get("visual_prompt"),
        "narration": shot.get("narration"),
        "subtitle": shot.get("subtitle"),
        "must_include": shot.get("intent_anchors") or [],
        "must_avoid": shot.get("must_avoid") or [],
        **spec,
    }


def plan_detail_durations(hints: list[float], total_duration: float) -> list[float]:
    """按候选的建议时长等比缩放到用户选择的总时长，最后一镜吸收舍入误差。"""
    count = len(hints)
    if count == 0:
        return []
    safe = [max(3.0, float(value or 5.0)) for value in hints]
    scale = float(total_duration) / sum(safe)
    durations = [round(value * scale, 2) for value in safe]
    elapsed = sum(durations[:-1])
    durations[-1] = round(float(total_duration) - elapsed, 2)
    if durations[-1] < 1:
        raise ValueError("所选总时长无法分配到这些镜头，请减少镜头数或延长时长")
    return durations


def build_composed_prompts(
    shots: list[dict[str, Any]], style_ctx: dict[str, Any]
) -> dict[str, str]:
    """用真实组合逻辑渲染每镜最终 prompt，用于长度校验与前端预览。

    与生成时用的是同一个 compose_prompt，避免"预览一套、实际发另一套"。
    """
    from .consistency import compose_prompt

    return {
        str(shot.get("id")): compose_prompt(
            str(shot.get("visual_prompt") or ""), style_ctx, shot=shot, index=int(shot.get("order") or 1)
        )
        for shot in shots
    }
