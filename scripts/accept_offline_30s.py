# -*- coding: utf-8 -*-
"""30 秒离线西湖成片验收（2026-11 重写，走现行产品路径）。

旧版绑死的 tts.build_audio / Orchestrator.apply_audio_durations / 旧
VideoRenderer.render 签名在三线合并后已不存在（docs/2026-11-23.md 遗留问题 1）。
本版不再复刻内部管线，改为驱动与 smoke-test.ps1 完全相同的产品路径：
Orchestrator.create_task 起全流水线 → 轮询任务 → ffprobe 校验成片。

验收点：
  - 任务 status=completed，无 error
  - final.mp4 / subtitles.srt 落盘在数据目录 media/<task_id>/
  - 容器/视频/音轨时长与所选秒数偏差 <= 0.08s（成片按旁白实测收口）
  - SRT 最后一条结束不超过成片时长 + 0.05s
  - 配音 provider 如实写进报告（离线默认系统语音；装好 edge-tts 则为 edge）

数据目录隔离在 runtime-accept/，不污染 runtime/。任意工作目录可运行：

    python scripts/accept_offline_30s.py

标准输出一行 JSON 摘要；全部通过退出码 0。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 隔离数据目录必须在导入 config 之前生效（Settings 在导入时读环境变量）。
os.environ.setdefault("DATA_DIR", "runtime-accept")
os.environ.setdefault("DEFAULT_PROVIDER", "offline")
os.environ.setdefault("ASSET_PROVIDER", "offline")

from backend.app.config import Settings  # noqa: E402
from backend.app.models import TaskInput  # noqa: E402
from backend.app.services.orchestrator import Orchestrator  # noqa: E402
from backend.app.storage import Storage  # noqa: E402

DURATION = 30
WIDTH, HEIGHT, FPS = 720, 1280, 25
POLL_TIMEOUT_SECONDS = 240


def _probe_duration(path: Path, select: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", select,
            "-show_entries",
            "format=duration" if select == "a" else "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True, capture_output=True, check=False,
    )
    values = []
    for line in (result.stdout or "").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else 0.0


def _probe_container(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True, capture_output=True, check=False,
    )
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return 0.0


def _srt_last_end(path: Path) -> float:
    last = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "-->" not in line:
            continue
        stamp = line.split("-->", 1)[1].strip().split()[0]
        hours, minutes, rest = stamp.split(":")
        seconds, millis = rest.split(",")
        value = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000
        last = max(last, value)
    return round(last, 3)


def _fail(reason: str) -> int:
    print(json.dumps({"ok": False, "reason": reason}, ensure_ascii=False))
    return 1


def main() -> int:
    settings = Settings()
    storage = Storage(settings.db_path)
    orchestrator = Orchestrator(settings, storage)
    orchestrator.recover_interrupted_tasks()

    task_input = TaskInput.from_dict(
        {
            "city": "杭州",
            "landmark": "西湖",
            "theme": "西湖六景",
            "culture": "宋韵",
            "duration": DURATION,
            "style": "诗意纪实",
            "music_mood": "无音乐",
            "shot_count": 6,
            "asset_source": "offline",
            "output_resolution": f"{WIDTH}x{HEIGHT}",
            "fps": FPS,
            "include_subtitles": True,
            "include_ai_label": True,
        }
    )
    task = orchestrator.create_task(task_input)

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        current = storage.get_task(task.id)
        if current and current.get("status") in {"completed", "failed"}:
            break
        time.sleep(1.0)
    else:
        return _fail(f"任务 {task.id} 在 {POLL_TIMEOUT_SECONDS}s 内未结束")
    if not current or current.get("status") != "completed":
        return _fail(f"任务失败：{current.get('error') if current else '未知'}")

    final_path = settings.media_dir / task.id / "final.mp4"
    srt_path = settings.media_dir / task.id / "subtitles.srt"
    if not final_path.is_file() or not srt_path.is_file():
        return _fail(f"成片或字幕缺失：{final_path} / {srt_path}")

    target = float(current["input"]["duration"])
    container = _probe_container(final_path)
    video = _probe_duration(final_path, "v:0")
    audio_sec = _probe_duration(final_path, "a:0")
    srt_last = _srt_last_end(srt_path)
    audio = current.get("audio") or {}
    report = {
        "ok": True,
        "task_id": task.id,
        "path": str(final_path),
        "target_sec": target,
        "container_sec": round(container, 3),
        "video_sec": round(video, 3),
        "audio_sec": round(audio_sec, 3),
        "srt_last_end": srt_last,
        "voice_provider": audio.get("voice_provider", ""),
        "has_cues": bool(audio.get("cues")),
        "shot_count": len(current.get("shots") or []),
        "stages_completed": [
            event.get("stage")
            for event in (current.get("trace") or {}).get("events", [])
            if event.get("type") == "stage" and event.get("status") == "completed"
        ],
    }
    print(json.dumps(report, ensure_ascii=False))
    if not all(abs(value - target) <= 0.08 for value in (container, video, audio_sec)):
        return _fail(f"时长偏差超限：container={container} video={video} audio={audio_sec} target={target}")
    if srt_last > target + 0.05:
        return _fail(f"字幕超出成片：srt_last={srt_last} > target={target}")
    if not report["has_cues"]:
        return _fail("配音没有产出实测 cues，无法证明音画对齐")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
