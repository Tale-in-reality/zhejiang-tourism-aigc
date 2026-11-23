"""Search stock footage, keeping attribution and the evidence for location matches."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

CITIES = dict(
    zip(
        "杭州 宁波 温州 嘉兴 湖州 绍兴 金华 衢州 舟山 台州 丽水".split(),
        "Hangzhou Ningbo Wenzhou Jiaxing Huzhou Shaoxing Jinhua Quzhou Zhoushan Taizhou Lishui".split(),
    )
)
LICENSE = {
    "pexels": "https://www.pexels.com/license/",
    "pixabay": "https://pixabay.com/service/license-summary/",
}


def _get(url, headers):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "ZhejiangTourism/2.0", **headers}
        ),
        timeout=30,
    )


@lru_cache(maxsize=256)
def search(platform, key, base, query, kind, day):
    video = kind == "video"
    headers = {}
    if platform == "pixabay":
        endpoint = "https://pixabay.com/api/" + ("videos/" if video else "")
        params = {"key": key, "q": query[:100], "safesearch": "true", "per_page": 30}
        if not video:
            params["image_type"] = "photo"
    else:
        endpoint = base.removesuffix("/v1") + (
            "/videos/search" if video else "/v1/search"
        )
        params = {"query": query, "per_page": 30}
        headers["Authorization"] = key
    try:
        with _get(endpoint + "?" + urllib.parse.urlencode(params), headers) as response:
            data = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            f"{platform} 检索失败 ({getattr(exc, 'code', type(exc).__name__)})"
        ) from None
    return data.get(
        "hits" if platform == "pixabay" else "videos" if video else "photos", []
    )


SHOT_TERMS = {
    "船": ("boat", "cruise"),
    "舟": ("boat",),
    "桥": ("bridge",),
    "塔": ("pagoda", "tower"),
    "亭": ("pavilion",),
    "柳": ("willow",),
    "堤": ("causeway",),
    "花": ("flower", "blossom"),
    "山": ("mountain", "hill"),
    "日落": ("sunset",),
    "夜": ("night",),
    "街": ("street",),
    "航拍": ("aerial", "drone"),
    "荷": ("lotus",),
    "倒影": ("reflection",),
}


def _candidate(platform, raw, kind, size, duration, anchors, city, query, terms=()):
    video = kind == "video"
    if platform == "pexels":
        page = raw.get("url", "")
        author = (
            (raw.get("user") or {}).get("name") if video else raw.get("photographer")
        )
        variants = (
            [
                (f.get("width"), f.get("height"), f.get("link"))
                for f in raw.get("video_files", [])
                if f.get("file_type") == "video/mp4"
            ]
            if video
            else [
                (raw.get("width"), raw.get("height"), raw.get("src", {}).get("large2x"))
            ]
        )
    else:
        page, author = raw.get("pageURL", ""), raw.get("user")
        variants = (
            [
                (f.get("width"), f.get("height"), f.get("url"))
                for f in raw.get("videos", {}).values()
            ]
            if video
            else [
                (
                    raw.get("imageWidth"),
                    raw.get("imageHeight"),
                    raw.get("largeImageURL"),
                )
            ]
        )
    variants = [
        (w, h, url)
        for w, h, url in variants
        if w and h and url and url.startswith("https://")
    ]
    if not variants or not page or not raw.get("id"):
        return None
    width, height = size

    def quality(v):
        w, h, _ = v
        return (
            abs(math.log(w / h / (width / height)))
            + abs(math.log(max(w / width, h / height))) * 0.16
        )

    w, h, url = min(variants, key=quality)
    tags = raw.get("tags") or ""
    if isinstance(tags, list):
        tags = " ".join(str(t) for t in tags)
    evidence = re.sub(
        r"[-_]+",
        " ",
        urllib.parse.unquote(" ".join([page, tags, raw.get("alt") or ""])),
    ).casefold()
    matches = [word for word in anchors if word and word.casefold() in evidence]
    city_match = bool(city and city.casefold() in evidence)
    if (
        raw.get("is_ai_generated") is True
        or raw.get("ai_generated") is True
        or "ai generated" in evidence
    ):
        return None
    tier = 0 if matches and city_match else 1 if matches else 2 if city_match else 3
    label = (
        ("地标线索：" + matches[0])
        if matches
        else ("城市氛围：" + city if city_match else "氛围素材，地点待核实")
    )
    hits = sum(word in evidence for word in terms)
    score = (
        tier,
        -hits,
        max(0, duration - raw.get("duration", 0)) if video else 0,
        quality((w, h, url)),
    )
    return score, dict(
        id=f"{platform}_{kind}_{raw['id']}",
        source=platform,
        page_url=page,
        author=author or platform,
        license=LICENSE[platform],
        name=label,
        usage_scope="按素材页和平台许可使用",
        is_ai_generated=False,
        media_type=kind,
        location_evidence=label,
        download=url,
        search_query=query,
    )


class StockProvider:
    def __init__(self, services, vision=None, location_check=True, max_checks=2):
        self.services = [service for service in services if service[1]]
        # 地域校验（可选）：搜"西湖"极可能返回别处的湖。有视觉模型时抽 1 帧问
        # "是否与目标城市/地标特征相符"，不符就换下一个候选。未配置模型时如实
        # 记为 unconfigured，不冒充已验证。
        self.vision = vision
        self.location_check = bool(location_check)
        self.max_checks = max(0, int(max_checks))

    def _verify_location(self, path, city, landmark):
        """抽 1 帧做地点特征判定。返回 (status, note)。"""
        from ..services import semantic_qa

        try:
            frames = semantic_qa.sample_frames(Path(path), 1, Path(path).parent / "stock_checks")
            instruction = (
                f"这张图/视频帧来自图库检索，候选地点是「{city}{landmark}」。"
                "请只输出 JSON：{\"location_plausible\":true/false,\"note\":\"一句话依据\"}。"
                "判断画面中的自然与建筑特征是否与这个地点相符；"
                "画面只拍到天空、水面纹理或无法判断时填 true。"
            )
            raw = self.vision.analyze(frames, instruction, max_tokens=200)
            plausible = raw.get("location_plausible")
            if isinstance(plausible, str):
                plausible = plausible.strip().lower() in {"true", "1", "yes", "是"}
            note = str(raw.get("note") or "").strip()[:160]
            return ("verified" if plausible is not False else "rejected"), note
        except Exception as exc:  # noqa: BLE001 - 校验失败不阻断取材，如实记录
            return "error", str(exc)[:160]

    def create_or_find(
        self, task_id, shot, output_dir, theme_color, accent, style_ctx=None
    ):
        ctx = style_ctx or {}
        w, h = map(int, ctx.get("output_resolution", "720x1280").split("x"))
        city = CITIES.get(shot.get("city"), shot.get("city", ""))
        landmark = shot.get("landmark", "")
        aliases = [
            s.strip() for s in shot.get("search_query", "").split(";") if s.strip()
        ]
        anchors = [landmark, *shot.get("landmark_aliases", [])]
        anchors = [a.strip() for a in anchors if a.strip()]
        if city:
            anchors += [
                a.casefold().removesuffix(" " + city.casefold()).strip()
                for a in anchors
                if a.casefold().endswith(" " + city.casefold())
            ]
        detail = next((s for s in aliases if s.isascii()), "")
        queries = list(
            dict.fromkeys(
                filter(
                    None,
                    [
                        (" ".join(filter(None, [detail, city]))).strip()
                        if detail
                        else "",
                        " ".join(
                            filter(
                                None,
                                [
                                    next(
                                        (a for a in anchors if a and a.isascii()),
                                        landmark,
                                    ),
                                    city,
                                ],
                            )
                        ),
                        city,
                        detail or shot.get("title"),
                    ],
                )
            )
        )
        prompt = (shot.get("title") or "") + " " + (shot.get("visual_prompt") or "")
        terms = {
            word
            for name, words in SHOT_TERMS.items()
            if name in prompt
            for word in words
        }
        used = ctx.get("stock_ids", set())
        lock = ctx.get("stock_lock") or threading.Lock()
        errors = []
        checks = 0
        output_dir.mkdir(exist_ok=True, parents=True)
        for kind in ("video", "image"):
            ranked = {}
            for query in queries:
                for platform, key, base in self.services:
                    try:
                        rows = search(
                            platform, key, base, query, kind, int(time.time() // 86400)
                        )
                    except RuntimeError as exc:
                        errors.append(str(exc))
                        continue
                    for raw in rows:
                        candidate = _candidate(
                            platform,
                            raw,
                            kind,
                            (w, h),
                            float(shot.get("duration_sec", 0)),
                            anchors,
                            city,
                            query,
                            terms,
                        )
                        if candidate:
                            ranked.setdefault(candidate[1]["id"], candidate)
                if any(
                    score[0] == 0 and a["id"] not in used
                    for score, a in ranked.values()
                ):
                    break
            for _, asset in sorted(ranked.values(), key=lambda entry: entry[0]):
                uid = asset["id"]
                with lock:
                    if uid in used:
                        continue
                    used.add(uid)
                path = output_dir / (
                    shot["id"] + "-" + uid + (".mp4" if kind == "video" else ".jpg")
                )
                tmp = path.with_suffix(".part")
                try:
                    with (
                        _get(asset.pop("download"), {}) as response,
                        tmp.open("wb") as target,
                    ):
                        shutil.copyfileobj(response, target)
                    probe = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height",
                            "-of",
                            "csv=p=0",
                            str(tmp),
                        ],
                        capture_output=True,
                        timeout=20,
                    )
                    if probe.returncode or not probe.stdout.strip():
                        raise ValueError("无法解码素材")
                    tmp.replace(path)
                except Exception as exc:
                    tmp.unlink(missing_ok=True)
                    with lock:
                        used.discard(uid)
                    errors.append(f"{uid}: {type(exc).__name__}")
                    continue
                location = {"status": "unconfigured", "note": "未配置视觉模型或已关闭地域校验"}
                if self.vision and self.location_check and checks < self.max_checks:
                    status, note = self._verify_location(path, city, landmark)
                    checks += 1
                    location = {"status": status, "note": note}
                    if status == "rejected":
                        path.unlink(missing_ok=True)
                        with lock:
                            used.discard(uid)
                        errors.append(
                            f"{uid}: 地点特征与{landmark or city}不符（{note}）"
                        )
                        continue
                return dict(
                    asset,
                    external_id=uid,
                    id=f"{task_id}_{shot['id']}_{uid}",
                    url=f"/media/{task_id}/assets/{path.name}",
                    local_path=str(path),
                    location_verification=location,
                )
        raise RuntimeError(
            "素材库没有可用的独立镜头。" + " / ".join(dict.fromkeys(errors))[:400]
        )
