"""候选池：一次生成若干"大方向镜头"（默认镜头数 ×2.5），用户自由选择并排序。

为什么单独一层：直接让模型写分镜时，画面描述是"结构正确"的副产品，一句话就交差。
候选池把"想拍什么"先摊开成互斥的视觉方向，用户用选择表达意图；被选中的
候选带着 imagery 锚点进入详述阶段，锚点强制继承——这是"生成的是想要的"的起点。

本模块只做三件事：
1. 拼候选生成指令（含配额、方向互斥、锚点可拍摄、事实溯源四条硬约束）；
2. 归一化与校验（不合格时把具体原因回灌重写，最多 2 轮）；
3. 把用户选择落成有序列表（配额不足/缺开场收束只提醒，不阻断）。
"""

from __future__ import annotations

import json
from typing import Any

from ..models import ShotCandidate

# 候选总数按用户选的镜头数放大：默认 2.5 倍，避免"想拍 8 镜却只给 10 个"。
# 下限给 target+2 个备选，上限 30 个（再多就该拆成两支片子了）。
CANDIDATE_MULTIPLIER = 2.5
MIN_CANDIDATES = 5
MAX_CANDIDATES = 30
# 角色配额的基准比例（对应 10 个时的 2/4/2/1/1），按总数等比换算后修正余数。
ROLE_WEIGHTS: dict[str, float] = {
    "opening": 0.2,
    "subject": 0.4,
    "detail": 0.2,
    "person": 0.1,
    "closing": 0.1,
}
ROLE_LABELS = {
    "opening": "开场（建立空间与时间）",
    "subject": "主体（最具辨识度的景观/建筑/场面）",
    "detail": "细节（材质、工艺、器物、局部）",
    "person": "人物（真实生活感，不摆拍、不正脸特写）",
    "closing": "收束（留白、远景或空镜）",
}


def target_shot_count(duration: int, shot_count: int) -> int:
    """用户显式选了镜头数就用它；自动模式按每镜约 5 秒切，夹在 2–12。"""
    if shot_count and int(shot_count) >= 2:
        return min(12, int(shot_count))
    duration = max(5, int(duration or 30))
    max_by_floor = max(1, duration // 4)
    return max(2, min(12, max_by_floor, max(2, round(duration / 5))))


def candidate_plan(target_shots: int) -> tuple[int, dict[str, int]]:
    """返回 (候选总数, 角色配额)。总数 = 镜头数 × 2.5，角色按权重等比分配。"""
    target = max(2, min(12, int(target_shots or 4)))
    total = int(round(target * CANDIDATE_MULTIPLIER))
    total = max(MIN_CANDIDATES, target + 2, min(MAX_CANDIDATES, total))
    roles = {
        role: max(1, int(round(total * weight)))
        for role, weight in ROLE_WEIGHTS.items()
    }
    # 修正取整误差：多退少补到主体（主体最容易多给也不伤叙事结构）
    while sum(roles.values()) > total:
        for role in ("subject", "detail", "opening", "person", "closing"):
            if sum(roles.values()) <= total:
                break
            if roles[role] > 1:
                roles[role] -= 1
    while sum(roles.values()) < total:
        roles["subject"] += 1
    return total, roles
ALLOWED_TIMES = ("清晨", "正午", "黄昏", "夜")
ALLOWED_SIZES = ("远景", "全景", "中景", "近景", "特写")
# 不可拍摄的形容词：锚点里出现这些词说明写的是"感觉"不是"物象"。
UNSHOOTABLE_TERMS = (
    "美景", "氛围", "震撼", "绝美", "治愈", "浪漫", "诗意", "意境",
    "仙气", "大片感", "出片", "打卡", "璀璨", "梦幻", "唯美",
)
MAX_IMAGERY = 4
MIN_IMAGERY = 2
CANDIDATE_REWRITE_ATTEMPTS = 2

CANDIDATES_SCHEMA = (
    '{"arc":{"opening":"...","development":"...","climax":"...","closing":"..."},'
    '"candidates":[{"id":"c1","role":"opening|subject|detail|person|closing",'
    '"title":"4-8字","hook":"一句话≤30字","imagery":["2-4个可拍摄的具体物象"],'
    '"place":"具体地点","time_of_day":"清晨|正午|黄昏|夜",'
    '"shot_size":"远景|全景|中景|近景|特写","motion":"画面内动态",'
    '"why":"为什么这一镜对主题有效≤40字","duration_hint":5}]}'
)


def build_candidates_instruction(
    *,
    city: str,
    landmark: str,
    theme: str,
    culture: str,
    festival: str,
    audience: str,
    style: str,
    custom_brief: str,
    knowledge_context: str,
    total: int = 0,
    roles: dict[str, int] | None = None,
) -> str:
    total = int(total or 10)
    roles = roles or candidate_plan(4)[1]
    quota = "、".join(
        f"{ROLE_LABELS[role]} {count} 个" for role, count in roles.items()
    )
    return (
        f"你是浙江文旅竖屏短片的分镜策划。为给定主题生成 {total} 个「可独立成镜」的大方向镜头，"
        "供用户挑选并排序。\n"
        f"用户输入：城市={city or '未指定'}；核心地标={landmark or '未指定'}；"
        f"主题={theme}；文化线索={culture or '未指定'}；节庆={festival or '未指定'}；"
        f"受众={audience or '未指定'}；叙事风格={style or '未指定'}；"
        f"补充要求={custom_brief or '无'}。\n"
        "先给一句叙事骨架 arc（开场/发展/高潮/收束各一句），再展开候选。\n"
        f"结构配额（必须满足）：{quota}。\n"
        f"方向互斥（最重要）：{total} 个候选必须是 {total} 个不同的「看什么 + 怎么看 + 什么时候看」；"
        "不允许把同一景观换形容词重复提交；任意两个候选不得同时满足"
        "同一主体 + 同一景别 + 同一时段。\n"
        f"锚点要求：每个候选给 {MIN_IMAGERY}–{MAX_IMAGERY} 个视觉锚点，"
        "必须是能拍进画面的具体物象（例：「青灰石栏内侧的缺口」「酒坛封口的红布」），"
        "不允许「美景」「氛围感」「震撼」这类不可拍摄的词。\n"
        "事实约束：地点、年代、数字、工艺名必须来自下方资料；资料没有的细节不得编造；"
        "place 必须落在用户所选城市。\n"
        "时长：每镜 duration_hint 取 3–8 秒。\n"
        f"{knowledge_context}\n"
        "只输出 JSON，不要 Markdown，不要解释。JSON 结构必须符合："
        f"{CANDIDATES_SCHEMA}"
    )


def normalize_candidates(payload: dict[str, Any]) -> tuple[dict[str, str], list[ShotCandidate]]:
    """解析模型输出为候选对象；字段缺失按默认值补齐（校验交给 validate_candidates）。"""
    arc_raw = payload.get("arc") if isinstance(payload.get("arc"), dict) else {}
    arc = {str(key): str(value).strip() for key, value in arc_raw.items() if str(value).strip()}
    items = payload.get("candidates")
    if not isinstance(items, list):
        raise RuntimeError("候选池响应缺少 candidates 数组")
    candidates: list[ShotCandidate] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        imagery = [
            str(anchor).strip()
            for anchor in (item.get("imagery") or [])
            if str(anchor).strip()
        ]
        role = str(item.get("role") or "").strip().lower()
        try:
            duration = float(item.get("duration_hint") or 5)
        except (TypeError, ValueError):
            duration = 5.0
        candidates.append(
            ShotCandidate(
                id=str(item.get("id") or f"c{index}"),
                role=role,
                title=str(item.get("title") or "").strip()[:20],
                hook=str(item.get("hook") or "").strip()[:60],
                imagery=imagery[:MAX_IMAGERY],
                entity_id=str(item.get("entity_id") or "").strip(),
                place=str(item.get("place") or "").strip()[:60],
                time_of_day=str(item.get("time_of_day") or "").strip(),
                shot_size=str(item.get("shot_size") or "").strip(),
                motion=str(item.get("motion") or "").strip()[:80],
                why=str(item.get("why") or "").strip()[:80],
                duration_hint=min(8.0, max(3.0, duration)),
            )
        )
    return arc, candidates


def validate_candidates(
    candidates: list[ShotCandidate],
    *,
    city: str = "",
    expected_total: int = 0,
    roles: dict[str, int] | None = None,
) -> list[str]:
    """返回问题列表；空列表代表合格。

    校验口径与提示词里的硬约束一一对应，任何一条不过都必须重写：
    数量与配额、方向互斥、锚点可拍摄、字段非空、地点在所选城市。
    expected_total/roles 由 candidate_plan 算出；不传时只做结构检查。
    """
    problems: list[str] = []
    total = int(expected_total or 0)
    if total and len(candidates) != total:
        problems.append(f"候选数量为 {len(candidates)}，必须正好 {total} 个")
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.role] = counts.get(candidate.role, 0) + 1
    if roles:
        for role, expected in roles.items():
            actual = counts.get(role, 0)
            if actual != expected:
                problems.append(f"{ROLE_LABELS.get(role, role)} 应为 {expected} 个，实际 {actual} 个")
        unknown_roles = sorted(set(counts) - set(roles))
    else:
        unknown_roles = sorted(set(counts) - set(ROLE_WEIGHTS))
    if unknown_roles:
        problems.append("出现未定义的角色：" + "、".join(unknown_roles))
    # 方向互斥：主体 + 景别 + 时段 三元组不得重复（用标题前 6 字粗粒度识别"同一主体"）
    seen: dict[tuple[str, str, str], str] = {}
    for candidate in candidates:
        subject_key = candidate.title[:6] or candidate.place
        key = (subject_key, candidate.shot_size, candidate.time_of_day)
        if key in seen:
            problems.append(
                f"候选 {seen[key]} 与 {candidate.id} 方向重复"
                f"（同一主体/景别/时段：{subject_key}/{candidate.shot_size or '未填'}/{candidate.time_of_day or '未填'}）"
            )
        else:
            seen[key] = candidate.id
    for candidate in candidates:
        label = candidate.id or candidate.title or "未命名候选"
        if not candidate.title:
            problems.append(f"{label} 缺少 title")
        if not candidate.hook:
            problems.append(f"{label} 缺少 hook（一句话说明这一镜讲什么）")
        if not (MIN_IMAGERY <= len(candidate.imagery) <= MAX_IMAGERY):
            problems.append(
                f"{label} 的视觉锚点应为 {MIN_IMAGERY}–{MAX_IMAGERY} 条，实际 {len(candidate.imagery)} 条"
            )
        for anchor in candidate.imagery:
            hit = next((term for term in UNSHOOTABLE_TERMS if term in anchor), "")
            if hit:
                problems.append(f"{label} 的锚点「{anchor}」含不可拍摄的词「{hit}」，请换成具体物象")
        if candidate.time_of_day and candidate.time_of_day not in ALLOWED_TIMES:
            problems.append(f"{label} 的 time_of_day 必须是 {'/'.join(ALLOWED_TIMES)}")
        if candidate.shot_size and candidate.shot_size not in ALLOWED_SIZES:
            problems.append(f"{label} 的 shot_size 必须是 {'/'.join(ALLOWED_SIZES)}")
        if city and candidate.place and city not in candidate.place:
            problems.append(f"{label} 的地点「{candidate.place}」不在所选城市 {city}")
    return problems


def candidate_rewrite_instruction(problems: list[str]) -> str:
    return (
        "上一版候选池不符合要求：\n- "
        + "\n- ".join(problems[:10])
        + "\n请修正后重新输出完整 JSON；结构配额、方向互斥、锚点可拍摄、地点在所选城市必须满足。"
        "只输出 JSON，不要解释。"
    )


def select_candidates(
    candidates: list[ShotCandidate], selection: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """校验并归一化用户选择。

    返回 (有序选择, 提醒列表)。硬错误抛 ValueError（未知候选/重复/数量超限），
    缺开场或缺收束只进提醒——用户自由选择优先，系统只给建议。
    """
    by_id = {candidate.id: candidate for candidate in candidates}
    if not by_id:
        raise ValueError("该任务没有候选池，无法提交选择")
    if not isinstance(selection, list) or not selection:
        raise ValueError("selection 不能为空")
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for index, item in enumerate(selection, start=1):
        if not isinstance(item, dict):
            raise ValueError("selection 的每一项必须是对象")
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("selection 缺少 candidate_id")
        if candidate_id not in by_id:
            raise ValueError(f"未知候选：{candidate_id}")
        if candidate_id in seen:
            raise ValueError(f"候选重复选择：{candidate_id}")
        seen.add(candidate_id)
        order = item.get("order")
        try:
            order_value = int(order) if order is not None else index
        except (TypeError, ValueError) as exc:
            raise ValueError(f"候选 {candidate_id} 的 order 不是整数") from exc
        ordered.append({"candidate_id": candidate_id, "order": order_value if order_value >= 1 else index})
    max_select = min(12, len(by_id))
    if not 2 <= len(ordered) <= max_select:
        raise ValueError(f"选择数量必须在 2–{max_select} 个之间")
    ordered.sort(key=lambda row: row["order"])
    for position, row in enumerate(ordered, start=1):
        row["order"] = position
    roles = [by_id[row["candidate_id"]].role for row in ordered]
    warnings: list[str] = []
    if "opening" not in roles:
        warnings.append("没有选开场镜头，成片可能缺少空间建立")
    if "closing" not in roles:
        warnings.append("没有选收束镜头，结尾可能显得突兀")
    if "detail" not in roles:
        warnings.append("没有选细节镜头，节奏可能单调")
    return ordered, warnings


def selection_to_json(ordered: list[dict[str, Any]]) -> str:
    return json.dumps(ordered, ensure_ascii=False)
