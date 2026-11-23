"""语义验收：VLM 逐镜判定"画面是不是这个分镜要的、主体有没有跑、有没有乱入"。

与 `visual_qa.py` 的分工：
- visual_qa：像素层（亮度/对比度/清晰度/熵/跨镜漂移）——不需要模型，永远能跑；
- semantic_qa：语义层（图文匹配、主体一致、违规元素、地点可信）——需要 VLM。

语义层未配置模型时报告 `unconfigured`，绝不写成 `passed`。判定失败的镜头
由 orchestrator 的自动重生成回路处理（改 prompt + 换 seed，限次），
仍然失败则进入人工复核清单。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..providers.vision import VisionProvider

SEMANTIC_QA_VERSION = "semantic-acceptance-v1"

# 违规元素归一化白名单。未知取值原样保留但只算 warning，不阻断——
# 让模型能报告新类型，又不让一个自由发挥的字段把整条片子拦下来。
BLOCKING_VIOLATIONS = {
    "modern_intrusion": "现代物闯入（广告牌/汽车/电线/现代招牌等）",
    "garbled_text": "画面出现乱码文字或伪中文",
    "cartoon_style": "卡通化/插画化，偏离纪实影像",
    "unrelated_scene": "画面与本镜描述无关",
    "landmark_distortion": "地标结构明显变形",
    "subject_mismatch": "主体与全片锁定主体不一致",
    "face_artifact": "人脸/肢体畸形",
}
WARNING_VIOLATIONS = {
    "extra_people": "出现无依据的路人",
    "low_quality": "画质明显偏低",
    "watermark": "画面带水印或台标",
}

FRAME_FRACTIONS = (0.15, 0.5, 0.85)


def _find_binary(name: str) -> str:
    import os

    override = str(os.getenv("FFMPEG_BINARY" if name == "ffmpeg" else "FFPROBE_BINARY", "") or "").strip()
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    root = Path(__file__).resolve().parents[3]
    candidate = root / "tools" / "ffmpeg" / "bin" / (f"{name}.exe" if os.name == "nt" else name)
    return str(candidate) if candidate.is_file() else ""


def _probe_duration(path: Path) -> float:
    binary = _find_binary("ffprobe")
    if not binary:
        return 0.0
    result = subprocess.run(
        [binary, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=False,
    )
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return 0.0


def sample_frames(media_path: Path, count: int, workdir: Path) -> list[Path]:
    """从图片或视频抽出 count 张判定帧。

    视频按 15%/50%/85% 取帧：首尾避开转场与黑场，中间帧代表镜头主体。
    抽帧失败抛 RuntimeError——语义验收不能拿"没抽到帧"当通过。
    """
    media_path = Path(media_path)
    if not media_path.is_file():
        raise RuntimeError(f"待判定素材不存在：{media_path}")
    if media_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return [media_path]
    binary = _find_binary("ffmpeg")
    if not binary:
        raise RuntimeError("找不到 ffmpeg，无法从视频抽帧做语义验收")
    duration = _probe_duration(media_path)
    if duration <= 0:
        raise RuntimeError(f"无法读取视频时长：{media_path.name}")
    count = max(1, min(5, int(count)))
    workdir.mkdir(parents=True, exist_ok=True)
    fractions = FRAME_FRACTIONS if count == 3 else tuple(
        (index + 0.5) / count for index in range(count)
    )
    frames: list[Path] = []
    for index, fraction in enumerate(fractions, start=1):
        target = workdir / f"frame_{index}.jpg"
        result = subprocess.run(
            [binary, "-v", "error", "-y", "-ss", f"{duration * fraction:.3f}",
             "-i", str(media_path), "-frames:v", "1", "-q:v", "2", str(target)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0 or not target.is_file():
            raise RuntimeError(
                f"抽帧失败（{media_path.name} @ {fraction:.0%}）："
                f"{(result.stderr or '').strip()[:160]}"
            )
        frames.append(target)
    return frames


def build_shot_instruction(
    *,
    order: int,
    visual_prompt: str,
    narration: str,
    city: str,
    landmark: str,
    subject_lock: str,
    check_location: bool = True,
    intent_anchors: list[str] | None = None,
) -> str:
    """单镜判定指令。字段名固定，便于解析与人工复核对齐。"""
    location_line = (
        f"5) location_plausible：画面是否与「{city}{landmark}」这类地方的自然与建筑特征相符"
        "（仅靠画面可判断的部分；判断不了就填 true 并在 note 说明）。\n"
        if check_location and (city or landmark)
        else '5) location_plausible：本片未指定地点，固定填 true。\n'
    )
    anchors = [str(anchor) for anchor in (intent_anchors or []) if str(anchor).strip()]
    anchors_line = (
        "7) present_anchors：从下列意图锚点中选出画面里确实出现的，逐字放进数组；"
        "没有出现的不要列：" + "、".join(anchors) + "\n"
        if anchors
        else ""
    )
    return (
        f"这是文旅短片第 {order} 镜的三帧（按时间顺序）。请逐项判定并只输出 JSON：\n"
        f"镜头画面要求：{visual_prompt}\n"
        f"该镜旁白：{narration}\n"
        f"全片主体锁定：{subject_lock}\n"
        "判定项：\n"
        "1) matches_shot：三帧画面是否就是这个镜头要求的内容（true/false）；\n"
        "2) subject_consistent：三帧之间主体是否一致、未变形（1-5 整数，5=完全一致）；\n"
        "3) reference_similarity：无参考图时填 0；有参考图时三帧主体与参考图的相似度（1-5）；\n"
        "4) violations：数组，从下列取值中选（没有就填空数组 []）：\n"
        f"   {', '.join(BLOCKING_VIOLATIONS)}\n"
        f"   以下只算提醒：{', '.join(WARNING_VIOLATIONS)}\n"
        f"{location_line}"
        f"{anchors_line}"
        "8) note：一句话说明判据（中文，40 字内）。\n"
        '输出格式：{"matches_shot":true,"subject_consistent":5,"reference_similarity":0,'
        '"violations":[],"location_plausible":true,"present_anchors":[],"note":""}'
    )


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_score(value: Any, default: float = 3.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(1.0, min(5.0, score))


def normalize_verdict(
    raw: dict[str, Any],
    *,
    min_subject_score: float = 3.0,
    check_location: bool = True,
    intent_anchors: list[str] | None = None,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """把模型返回归一化成固定结构，并给出 passed 与原因。

    判定口径（全部满足才算通过）：
    - matches_shot 为真；
    - subject_consistent >= 阈值；
    - 没有 BLOCKING_VIOLATIONS 里的违规；
    - 开启地点检查时 location_plausible 为真。
    未知违规词保留在 warnings，不阻断（避免模型自由发挥把整片拦死）。
    """
    raw = raw if isinstance(raw, dict) else {}
    violations_raw = raw.get("violations")
    violations: list[str] = []
    if isinstance(violations_raw, list):
        for item in violations_raw:
            text = str(item).strip()
            if text and text not in violations:
                violations.append(text)
    blocking = [item for item in violations if item in BLOCKING_VIOLATIONS]
    warnings = [item for item in violations if item not in BLOCKING_VIOLATIONS]
    matches = _as_bool(raw.get("matches_shot"), default=True)
    subject = _as_score(raw.get("subject_consistent"))
    location = _as_bool(raw.get("location_plausible"), default=True)
    reference = _as_score(raw.get("reference_similarity"), default=0.0)
    # 意图覆盖率：必现锚点有多少真的出现在画面里。没传锚点时按 1.0 处理（不参与判定）。
    anchors = [str(value) for value in (intent_anchors or []) if str(value).strip()]
    present_raw = raw.get("present_anchors")
    present = [str(value).strip() for value in present_raw] if isinstance(present_raw, list) else []
    present = [item for item in present if item]
    if anchors:
        matched = [
            anchor for anchor in anchors
            if any(anchor in item or item in anchor for item in present)
        ]
        missing_anchors = [anchor for anchor in anchors if anchor not in matched]
        coverage = round(len(matched) / len(anchors), 4)
    else:
        missing_anchors = []
        coverage = 1.0
    reasons: list[str] = []
    if not matches:
        reasons.append("画面与分镜描述不符")
    if subject < float(min_subject_score):
        reasons.append(f"主体一致性 {subject:g} 低于阈值 {float(min_subject_score):g}")
    if blocking:
        reasons.append("违规元素：" + "、".join(BLOCKING_VIOLATIONS.get(item, item) for item in blocking))
    if check_location and not location:
        reasons.append("画面地点特征与目标城市/地标不符")
    if anchors and coverage < float(min_coverage):
        reasons.append(
            f"意图锚点覆盖 {coverage:.0%} 低于 {float(min_coverage):.0%}，缺："
            + "、".join(missing_anchors[:4])
        )
    return {
        "matches_shot": matches,
        "subject_consistent": subject,
        "reference_similarity": reference,
        "violations": violations,
        "blocking_violations": blocking,
        "warnings": warnings,
        "location_plausible": location,
        "intent_coverage": coverage,
        "present_anchors": present,
        "missing_anchors": missing_anchors,
        "note": str(raw.get("note") or "").strip()[:200],
        "passed": not reasons,
        "reasons": reasons,
    }


def judge_shot(
    vision: VisionProvider,
    frames: list[Path],
    *,
    order: int,
    visual_prompt: str,
    narration: str,
    city: str,
    landmark: str,
    subject_lock: str,
    min_subject_score: float = 3.0,
    check_location: bool = True,
    intent_anchors: list[str] | None = None,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """一次判定；网络/解析异常包装成 RuntimeError 由上层记 failed。"""
    instruction = build_shot_instruction(
        order=order,
        visual_prompt=visual_prompt,
        narration=narration,
        city=city,
        landmark=landmark,
        subject_lock=subject_lock,
        check_location=check_location,
        intent_anchors=intent_anchors,
    )
    raw = vision.analyze(frames, instruction)
    verdict = normalize_verdict(
        raw,
        min_subject_score=min_subject_score,
        check_location=check_location,
        intent_anchors=intent_anchors,
        min_coverage=min_coverage,
    )
    verdict["raw"] = raw
    return verdict


def regeneration_clause(verdict: dict[str, Any]) -> str:
    """把判定失败原因改写成可注入 prompt 的返工要求。

    写具体的"不要什么/必须出现什么"而不是"注意质量"：生成模型对具体条目更服从。
    缺锚点时点名要求出现——这是"定向修正"与"整段重写"的区别。
    """
    parts: list[str] = []
    missing = [str(item) for item in (verdict.get("missing_anchors") or []) if str(item)]
    if missing:
        parts.append("画面必须明确出现：" + "、".join(missing[:4]))
    for item in verdict.get("blocking_violations") or []:
        label = BLOCKING_VIOLATIONS.get(item, item)
        parts.append(f"严禁出现{label}")
    if not verdict.get("matches_shot", True):
        parts.append("画面必须直接呈现本镜描述的主体与动作，不要换成其他场景")
    if float(verdict.get("subject_consistent", 5)) < 3:
        parts.append("同一主体在整段画面中保持外形、材质与比例不变")
    if not verdict.get("location_plausible", True):
        parts.append("环境必须是目标地点的真实自然与建筑特征，不要出现其他城市的标志物")
    if not parts:
        parts.append("严格贴合本镜描述，主体与场景保持不变")
    note = str(verdict.get("note") or "").strip()
    suffix = f"（审片意见：{note}）" if note else ""
    return "返工要求：" + "；".join(parts) + "。" + suffix


def build_report(
    *,
    status: str,
    provider: str = "",
    model: str = "",
    shots: list[dict[str, Any]] | None = None,
    regenerated: list[dict[str, Any]] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    shots = shots or []
    blocking = [
        str(item.get("shot_id"))
        for item in shots
        if item.get("verdict") and not item["verdict"].get("passed", True)
    ]
    return {
        "version": SEMANTIC_QA_VERSION,
        "status": status,  # checked / unconfigured / skipped / failed
        "provider": provider,
        "model": model,
        "check_location": True,
        "shots": shots,
        "regenerated": regenerated or [],
        "blocking_shots": blocking,
        "passed": status == "checked" and not blocking,
        "manual_review_required": bool(blocking),
        "reason": reason,
    }


def merge_into_visual_qa(visual_qa: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    """把语义报告嵌进 visual_qa，manifest 里只有一份视觉验收文件。"""
    merged = dict(visual_qa or {})
    merged["semantic"] = semantic
    return merged


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
