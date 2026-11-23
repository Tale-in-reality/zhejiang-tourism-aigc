"""旁白文案的"去 AI 味"质检：词表扫描与命中报告。

词表外置在 `copy_banned_terms.json`（与本模块同目录），评测脚本
`scripts/copy_quality_eval.py` 和 llm.py 的改写回路共用同一份数据，
保证"评测口径 = 拦截口径"。词表只收录高置信度套话，宁可漏收不可误伤：
命中会触发整段旁白改写，误伤一次好文案比漏放一句套话代价大。

扫描对象是"给人听/看的文字"（标题、摘要、CTA、旁白、字幕）与"给模型
看的文字"（visual_prompt）分开报告——后者命中只记录不强制改写，因为
生成 prompt 里的风格词由 consistency 层约束管理。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TERMS_PATH = Path(__file__).with_name("copy_banned_terms.json")

# 脚本级文字字段（命中触发改写）
SCRIPT_TEXT_FIELDS: tuple[str, ...] = ("title", "summary", "cta")
# 镜头级文字字段（命中触发改写）
SHOT_TEXT_FIELDS: tuple[str, ...] = ("narration", "subtitle")
# 镜头级生成字段（命中只记录，供改写时顺带处理）
SHOT_PROMPT_FIELDS: tuple[str, ...] = ("visual_prompt",)

_cache: dict[str, Any] | None = None


def load_banned_terms(path: Path | None = None) -> dict[str, Any]:
    """读取词表并编译正则；文件缺失或损坏时退回空表（质检降级不阻断）。"""
    global _cache
    source = path or TERMS_PATH
    if path is None and _cache is not None:
        return _cache
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 词表损坏时按空表处理
        data = {}
    words = [str(w).strip() for w in (data.get("words") or []) if str(w).strip()]
    patterns = []
    for item in data.get("patterns") or []:
        if isinstance(item, str):
            entry = {"pattern": item, "label": "AI 腔句式"}
        elif isinstance(item, dict):
            entry = {"pattern": str(item.get("pattern", "")).strip(), "label": str(item.get("label") or "AI 腔句式")}
        else:
            continue
        if not entry["pattern"]:
            continue
        try:
            entry["regex"] = re.compile(entry["pattern"])
        except re.error:
            continue
        patterns.append(entry)
    result = {
        "version": str(data.get("version") or "unknown"),
        "words": words,
        "patterns": patterns,
    }
    if path is None:
        _cache = result
    return result


def scan_text(text: str, terms: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """对一段文字做词表扫描，返回命中列表（含词、类别、位置与上下文摘录）。"""
    value = str(text or "")
    if not value.strip():
        return []
    terms = terms or load_banned_terms()
    hits: list[dict[str, Any]] = []
    for word in terms["words"]:
        start = 0
        while True:
            index = value.find(word, start)
            if index < 0:
                break
            hits.append(_hit(word, "AI 腔词", index, value))
            start = index + len(word)
    for entry in terms["patterns"]:
        for match in entry["regex"].finditer(value):
            hits.append(_hit(match.group(0), entry["label"], match.start(), value))
    return hits


def _hit(term: str, label: str, index: int, value: str) -> dict[str, Any]:
    left = max(0, index - 8)
    right = min(len(value), index + len(term) + 8)
    return {
        "term": term,
        "label": label,
        "position": index,
        "excerpt": value[left:right],
    }


def _scan_fields(fields: tuple[str, ...], payload: dict[str, Any], terms: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for field in fields:
        for hit in scan_text(payload.get(field), terms):
            hit["field"] = field
            hits.append(hit)
    return hits


def scan_script(result: dict[str, Any], terms: dict[str, Any] | None = None) -> dict[str, Any]:
    """对完整脚本 JSON 做扫描。

    返回结构：
    - ``hits``：触发改写的命中（title/summary/cta/narration/subtitle）
    - ``prompt_hits``：visual_prompt 里的命中（只记录）
    - ``ok``：hits 为空
    """
    terms = terms or load_banned_terms()
    if not isinstance(result, dict):
        return {"hits": [], "prompt_hits": [], "ok": True}
    hits = _scan_fields(SCRIPT_TEXT_FIELDS, result, terms)
    shots = result.get("shots") if isinstance(result.get("shots"), list) else []
    for shot in shots:
        if isinstance(shot, dict):
            hits.extend(_scan_fields(SHOT_TEXT_FIELDS, shot, terms))
    prompt_hits: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        if isinstance(shot, dict):
            for hit in _scan_fields(SHOT_PROMPT_FIELDS, shot, terms):
                hit["shot"] = index
                prompt_hits.append(hit)
    return {"hits": hits, "prompt_hits": prompt_hits, "ok": not hits}


def hit_terms(hits: list[dict[str, Any]]) -> list[str]:
    """去重后的命中词列表，用于改写指令与事件记录。"""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(str(hit.get("term", "")), None)
    return [term for term in seen if term]


def format_hit_fields(hits: list[dict[str, Any]], limit: int = 6) -> str:
    """把命中压成「字段:摘录」的可读串，拼进改写指令。"""
    parts = []
    for hit in hits[:limit]:
        field = hit.get("field") or f"shot_{hit.get('shot', '?')}.{hit.get('field')}"
        parts.append(f"{field}「{hit.get('excerpt', '')}」")
    return "；".join(parts)
