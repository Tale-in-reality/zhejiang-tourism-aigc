"""OpenAI 兼容多模态视觉验收 provider（VLM）。

为什么需要这一层：`visual_qa.py` 的像素指标只能判断"画面是不是黑的/糊的/漂移"，
判断不了"画面是不是这个分镜要的内容、主体还是不是同一个、有没有现代物闯入"。
语义层必须由视觉语言模型给出，这里接的是最通用的 OpenAI 兼容多模态协议
（`/chat/completions` + `image_url` 数组），GLM-4V / Qwen-VL / 豆包视觉 /
任何自建兼容网关都能直接用，换模型只改环境变量。

未配置 `VISION_MODEL` 时 `build_vision_provider` 返回 None，上层必须把语义验收
记为 `unconfigured`——项目既有约定：不把"没检查"记成"通过"。
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..services.cost import provider_call, record_response

PROVIDER_NAME = "openai-compatible-vision"
VISION_SYSTEM_PROMPT = "你是严格的文旅短视频画面审片人，只输出合法 JSON。"
DEFAULT_TIMEOUT_SECONDS = 60
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _image_data_url(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _strip_json_fence(content: str) -> str:
    return _JSON_FENCE.sub("", str(content or "").strip())


def parse_json_object(content: str) -> dict[str, Any]:
    """解析模型返回的 JSON 对象；剥 Markdown 壳，非对象直接报错。"""
    result = json.loads(_strip_json_fence(content))
    if not isinstance(result, dict):
        raise ValueError("vision response is not a JSON object")
    return result


def build_vision_message(text: str, frames: list[str]) -> list[dict[str, Any]]:
    """组装 OpenAI 兼容多模态 user 消息：文字在前，图依次附后。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in frames:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


class VisionProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def analyze(
        self,
        frames: list[Path],
        instruction: str,
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        """送 1–5 张帧 + 判定指令，返回解析后的 JSON 判定。

        只发一次、不自动重试：视觉调用按量计费，重试由上层的"语义不过→重生成"
        回路承担，避免同一判定扣两次费。
        """
        if not frames:
            raise ValueError("vision analyze requires at least one frame")
        missing = [str(frame) for frame in frames if not Path(frame).is_file()]
        if missing:
            raise RuntimeError(f"待判定帧不存在：{', '.join(missing)}")
        urls = [_image_data_url(Path(frame)) for frame in frames]
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                *build_vision_message(instruction, urls),
            ],
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
        try:
            with provider_call("vision", self.name, self.model) as billing:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    record_response(billing, response)
                    payload = json.loads(response.read().decode("utf-8"))
                    record_response(billing, response, payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"视觉判定被拒绝：HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"视觉判定请求没有完成，未自动重试：{exc.reason}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("视觉判定响应缺少 choices[0].message.content") from exc
        # 部分网关把图片理解结果放在 content 数组里，取其中的 text 段
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        return parse_json_object(str(content or ""))


def build_vision_provider(settings) -> VisionProvider | None:
    """按配置构造 provider；VISION_MODEL 为空即视为未配置（返回 None）。

    base/key 的回落顺序：VISION_* 显式配置 → LLM_*（同一套 OpenAI 兼容凭据）。
    模型名没有合理回落：视觉模型必须显式指定，猜一个名字只会得到 404。
    """
    model = str(getattr(settings, "vision_model", "") or "").strip()
    if not model:
        return None
    base_url = str(
        getattr(settings, "vision_base_url", "")
        or getattr(settings, "llm_base_url", "")
        or ""
    ).strip()
    api_key = str(
        getattr(settings, "vision_api_key", "")
        or getattr(settings, "llm_api_key", "")
        or ""
    ).strip()
    if not base_url:
        return None
    timeout = int(getattr(settings, "vision_timeout_seconds", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS)
    return VisionProvider(base_url, api_key, model, timeout=timeout)
