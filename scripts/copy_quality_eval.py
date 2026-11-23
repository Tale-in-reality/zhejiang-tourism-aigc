# -*- coding: utf-8 -*-
"""文本质量评测：AI 腔词表扫描 + 画面 prompt 五要素完备率 + 可选 LLM judge。

阶段一验收口径。词表与 llm.py 改写回路共用同一份
（backend/app/services/copy_quality.py 读 copy_banned_terms.json），
保证"拦截口径 = 评测口径"，不会出现评测比拦截更严或更松。

达标线（阶段一 Goal）：
  1. AI 腔词表命中 = 0
  2. 画面 prompt 五要素完备率 >= 95%
  3. LLM judge 三维（人味/具体性/地方特色）评分报告——需 LLM_API_KEY，缺则跳过

用法：
  python scripts/copy_quality_eval.py                          # golden 5 主题：生成→精修→评测
  python scripts/copy_quality_eval.py --themes 2               # 只跑前 2 个主题
  python scripts/copy_quality_eval.py --input script.json      # 评测任意脚本 JSON
  python scripts/copy_quality_eval.py --task-id task_x --data-dir runtime
  python scripts/copy_quality_eval.py --baseline old.json      # 新旧对比

报告写 runtime-eval/copy_quality_report.json 并打印摘要与达标判定。
生成走 .env 配置的文本 provider（无 key 时自动退回离线模板，报告如实标注）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import PROJECT_ROOT, settings  # noqa: E402
from backend.app.providers.llm import OpenAICompatibleLLMProvider  # noqa: E402
from backend.app.providers.offline import OfflineLLMProvider  # noqa: E402
from backend.app.services import copy_quality  # noqa: E402
from backend.app.services.consistency import missing_visual_prompt_elements  # noqa: E402

# 团队方案文档里的 5 个基准样例主题（杭州/绍兴用于开发，全部用于回归）。
GOLDEN_THEMES: list[dict[str, object]] = [
    {
        "city": "杭州", "landmark": "西湖", "theme": "西湖春游",
        "culture": "西湖文化景观", "festival": "春茶季", "audience": "家庭游客",
        "duration": 30, "style": "诗意纪实",
    },
    {
        "city": "绍兴", "landmark": "仓桥直街", "theme": "黄酒文化",
        "culture": "黄酒酿造技艺", "festival": "黄酒节", "audience": "年轻游客",
        "duration": 30, "style": "城市漫游",
    },
    {
        "city": "嘉兴", "landmark": "南湖", "theme": "红船红色旅游",
        "culture": "红船精神", "festival": "", "audience": "研学团体",
        "duration": 30, "style": "诗意纪实",
    },
    {
        "city": "丽水", "landmark": "古堰画乡", "theme": "古堰画乡写生之旅",
        "culture": "通济堰水利文化", "festival": "写生季", "audience": "文艺游客",
        "duration": 30, "style": "城市漫游",
    },
    {
        "city": "舟山", "landmark": "普陀山", "theme": "海岛日出",
        "culture": "观音文化", "festival": "", "audience": "休闲游客",
        "duration": 30, "style": "诗意纪实",
    },
]

FIVE_ELEMENT_TARGET = 0.95


def make_provider():
    """按 .env 配置取文本 provider；无 key 退回离线模板并让报告如实标注。"""
    model = getattr(settings, "llm_model", "")
    if settings.llm_base_url and settings.llm_api_key and model:
        return (
            OpenAICompatibleLLMProvider(settings.llm_base_url, settings.llm_api_key, model),
            True,
        )
    return OfflineLLMProvider(), False


def build_scripts(themes: list[dict[str, object]], provider) -> list[dict[str, object]]:
    """生成 → 精修 → 收集。精修失败不阻断，事件里如实记录。"""
    entries: list[dict[str, object]] = []
    for theme in themes:
        prompt = json.dumps(theme, ensure_ascii=False)
        context = dict(theme)
        result = provider.generate(prompt, context)
        data = result.data
        shots = data.get("shots") or []
        refine_event: dict[str, object] | None = None
        if shots and hasattr(provider, "refine_prompts"):
            try:
                outcome = provider.refine_prompts(shots, context)
                refined_map = {str(item["id"]): item for item in outcome.get("refined", [])}
                for shot in shots:
                    item = refined_map.get(str(shot.get("id")))
                    if item and str(item.get("visual_prompt", "")).strip():
                        shot["visual_prompt"] = item["visual_prompt"]
                refine_event = dict(outcome.get("event") or {})
            except Exception as exc:  # noqa: BLE001 - 精修失败保留原 prompt 评测
                refine_event = {"ok": False, "error": str(exc)}
        entries.append(
            {
                "theme": theme,
                "script": data,
                "refine_event": refine_event,
                "provider": result.provider,
                "model": result.model,
                "fallback_used": result.fallback_used,
            }
        )
    return entries


def evaluate_script(script: dict[str, object]) -> dict[str, object]:
    scan = copy_quality.scan_script(script)
    shots = script.get("shots") if isinstance(script.get("shots"), list) else []
    missing: dict[str, list[str]] = {}
    for shot in shots:
        if isinstance(shot, dict):
            key = str(shot.get("id") or len(missing) + 1)
            missing[key] = missing_visual_prompt_elements(str(shot.get("visual_prompt", "")))
    total = len(missing)
    passed = sum(1 for values in missing.values() if not values)
    return {
        "ai_tone": {
            "hits": len(scan["hits"]),
            "prompt_hits": len(scan["prompt_hits"]),
            "terms": copy_quality.hit_terms(scan["hits"]),
        },
        "five_element": {
            "total": total,
            "pass": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "missing": missing,
        },
    }


def judge_script(provider, script: dict[str, object]) -> dict[str, object]:
    """LLM judge 三维评分。离线模板没有 judge 意义，由调用方跳过。"""
    shots = script.get("shots") if isinstance(script.get("shots"), list) else []
    narration = str(script.get("narration") or "。".join(
        str(shot.get("narration", "")) for shot in shots if isinstance(shot, dict)
    ))
    messages = [
        {"role": "system", "content": "你是严格的短视频文案审稿人，只输出合法 JSON。"},
        {
            "role": "user",
            "content": (
                "给下面的文旅短视频旁白打分（1-5 整数）：\n"
                "human_likeness 人味：像人写的口播稿还是 AI 模板腔；\n"
                "specificity 具体性：有没有能拍进画面的具体名词细节；\n"
                "local_flavor 地方特色：是否体现当地地名、物产、工艺或气质。\n"
                '只输出 JSON：{"human_likeness":1,"specificity":1,"local_flavor":1,"notes":"一句话理由"}\n'
                f"旁白：{narration[:1200]}"
            ),
        },
    ]
    try:
        result = provider._chat(messages)  # noqa: SLF001 - 评测脚本内部复用同一调用通道
        return {
            "human_likeness": result.get("human_likeness"),
            "specificity": result.get("specificity"),
            "local_flavor": result.get("local_flavor"),
            "notes": result.get("notes"),
        }
    except Exception as exc:  # noqa: BLE001 - judge 失败不影响词表/五要素两项
        return {"error": str(exc)}


def load_script_from_file(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("input"), dict) and "shots" in data:
        # 任务 JSON：抽出脚本相关字段
        shots = data.get("shots") or []
        return {
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "cta": data.get("cta", ""),
            "narration": data.get("narration", ""),
            "shots": [
                {
                    "id": shot.get("id"),
                    "visual_prompt": shot.get("visual_prompt", ""),
                    "narration": shot.get("narration", ""),
                    "subtitle": shot.get("subtitle", ""),
                }
                for shot in shots
                if isinstance(shot, dict)
            ],
        }
    return data


def load_script_from_task(task_id: str, data_dir: Path) -> dict[str, object]:
    from backend.app.storage import Storage

    db_path = data_dir / "db" / "app.sqlite3"
    data = Storage(db_path).get_task(task_id)
    if not data:
        raise SystemExit(f"task not found: {task_id} (db: {db_path})")
    return load_script_from_file(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--themes", type=int, default=0, help="只跑前 N 个 golden 主题（0=全部）")
    parser.add_argument("--input", type=Path, help="评测指定脚本/任务 JSON")
    parser.add_argument("--task-id", help="评测 runtime 里已存在的任务")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir, help="任务数据目录")
    parser.add_argument("--baseline", type=Path, help="基线脚本 JSON，用于新旧对比")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM judge")
    args = parser.parse_args()

    provider, live = make_provider()
    entries: list[dict[str, object]] = []
    if args.input or args.task_id:
        script = (
            load_script_from_file(args.input)
            if args.input
            else load_script_from_task(args.task_id, Path(args.data_dir))
        )
        entries.append({"theme": {"source": str(args.input or args.task_id)}, "script": script, "provider": "input"})
    else:
        themes = GOLDEN_THEMES[: args.themes] if args.themes else GOLDEN_THEMES
        entries = build_scripts(themes, provider)

    for entry in entries:
        entry["evaluation"] = evaluate_script(entry["script"])
        if live and not args.no_judge:
            entry["judge"] = judge_script(provider, entry["script"])

    baseline_eval = evaluate_script(load_script_from_file(args.baseline)) if args.baseline else None

    word_hit_total = sum(entry["evaluation"]["ai_tone"]["hits"] for entry in entries)
    prompt_hit_total = sum(entry["evaluation"]["ai_tone"]["prompt_hits"] for entry in entries)
    rates = [
        entry["evaluation"]["five_element"]["pass_rate"]
        for entry in entries
        if entry["evaluation"]["five_element"]["total"]
    ]
    min_rate = min(rates) if rates else 0.0
    judge_scores = [
        entry["judge"]
        for entry in entries
        if isinstance(entry.get("judge"), dict) and "error" not in entry["judge"]
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai-compatible" if live else "offline-zhejiang-editor",
        "live_llm": live,
        "themes": entries,
        "baseline": baseline_eval,
        "summary": {
            "theme_count": len(entries),
            "ai_tone_hits_total": word_hit_total,
            "ai_tone_prompt_hits_total": prompt_hit_total,
            "five_element_pass_rate_min": min_rate,
            "judge_scored": len(judge_scores),
        },
        "gates": {
            "word_hits_zero": word_hit_total == 0,
            "five_element_ge_95": bool(rates) and min_rate >= FIVE_ELEMENT_TARGET,
        },
    }
    out_dir = PROJECT_ROOT / "runtime-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "copy_quality_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report: {out_path}")
    print(f"provider: {'live LLM' if live else 'OFFLINE 模板（judge 跳过，结果不代表真实链路）'}")
    print(f"AI 腔词表命中：{word_hit_total}（画面 prompt 命中 {prompt_hit_total}，只记录不改写）")
    print(f"五要素完备率（最低主题）：{min_rate:.0%}  目标 ≥ {FIVE_ELEMENT_TARGET:.0%}")
    if baseline_eval is not None:
        print(
            "新旧对比 — 基线词表命中："
            f"{baseline_eval['ai_tone']['hits']} → {word_hit_total}；"
            f"五要素 {baseline_eval['five_element']['pass_rate']:.0%} → "
            f"{min_rate:.0%}"
        )
    gates_ok = report["gates"]["word_hits_zero"] and (
        report["gates"]["five_element_ge_95"] or not rates
    )
    print("达标判定：" + ("PASS" if gates_ok else "FAIL"))
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
