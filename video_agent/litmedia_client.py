"""LitMedia 文生视频 HTTP 客户端与 LangChain Tool 封装。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool

from video_agent.logging_utils import debug_print
from video_agent.models import (
    CreativeParams,
    FrontendParams,
    SubmitImageToVideoInput,
    SubmitVideoGenerationInput,
)
from video_agent.settings import LITMEDIA_API_URL
LITMEDIA_IMAGE_VIDEO_API_URL = "https://litvideo-api.litmedia.ai/lit-video/do-image-video"


def generate_signature_params() -> dict[str, str]:
    """生成 LitMedia 接口所需动态签名字段。"""

    api_secret = (os.getenv("LITMEDIA_API_SECRET") or "").strip()
    fingerprint = (os.getenv("LITMEDIA_DEVICE_CODE") or "").strip()
    if not api_secret or not fingerprint:
        raise ValueError(
            "调用 LitMedia 需在环境中配置 LITMEDIA_API_SECRET 与 LITMEDIA_DEVICE_CODE。"
        )

    timestamp = str(int(time.time() * 1000))
    random_str = str(random.randint(0, 100_000_000))
    sha1_hex = hashlib.sha1(f"{timestamp}{random_str}{api_secret}".encode()).hexdigest()
    signature = hashlib.md5(sha1_hex.encode()).hexdigest().upper()
    sign = hashlib.sha1(
        f"{timestamp}{random_str}{api_secret}{signature}".encode()
    ).hexdigest().upper()

    return {
        "timeStamp": timestamp,
        "randomStr": random_str,
        "signature": signature,
        "fingerprint": fingerprint,
        "sign": sign,
    }


def build_litmedia_payload(
    creative_params: CreativeParams,
    frontend_params: FrontendParams,
) -> dict[str, str]:
    """合并 Agent 内容参数和前端 UI 参数，构建接口表单。"""

    if not creative_params.prompt:
        raise ValueError("缺少 prompt，无法提交生成。")

    signature_params = generate_signature_params()
    return {
        "video_model": frontend_params.video_model,
        "video_num": str(frontend_params.video_num),
        "prompt": creative_params.prompt,
        "open_filter": str(frontend_params.open_filter),
        "sound_effect_switch": str(frontend_params.sound_effect_switch),
        "ratio": frontend_params.ratio,
        "quality": frontend_params.quality,
        "duration": str(frontend_params.duration),
        "seed": frontend_params.seed,
        "negative_prompt": creative_params.negative_prompt or "",
        "is_feed": str(frontend_params.is_feed),
        **signature_params,
    }


def build_litmedia_i2v_payload(
    creative_params: CreativeParams,
    frontend_params: FrontendParams,
) -> dict[str, str]:
    """合并图生参数，构建图生接口表单。"""

    if not creative_params.source_image_url:
        raise ValueError("缺少 source_image_url，无法提交图生任务。")
    if not creative_params.prompt:
        raise ValueError("缺少 prompt，无法提交图生任务。")

    signature_params = generate_signature_params()
    return {
        "video_model": frontend_params.video_model,
        "video_num": str(frontend_params.video_num),
        "img_url": creative_params.source_image_url,
        "prompt": creative_params.prompt,
        "open_filter": str(frontend_params.open_filter),
        "sound_effect_switch": str(frontend_params.sound_effect_switch),
        "quality": frontend_params.quality,
        "duration": str(frontend_params.duration),
        "seed": frontend_params.seed,
        "negative_prompt": creative_params.negative_prompt or "",
        "is_feed": str(frontend_params.is_feed),
        **signature_params,
    }


def build_litmedia_headers(fingerprint: str) -> dict[str, str]:
    """构建 LitMedia 请求头。"""

    token = (os.getenv("LITMEDIA_TOKEN") or "").strip()
    if not token:
        raise ValueError("调用 LitMedia 需在环境中配置 LITMEDIA_TOKEN。")

    return {
        "accept": "application/json",
        "accept-language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
        "lang": "ZH-HANT",
        "monimaster-device-code": fingerprint,
        "monimaster-device-type": "3",
        "monimaster-token": token,
        "nation-code": "EN",
        "origin": "https://www.litmedia.ai",
        "referer": "https://www.litmedia.ai/",
        "timezone": "Asia/Shanghai",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
    }


def call_litmedia_text_video(
    creative_params: CreativeParams,
    frontend_params: FrontendParams,
) -> dict[str, Any]:
    """调用 LitMedia 文生视频接口。"""

    payload = build_litmedia_payload(creative_params, frontend_params)
    debug_print("LitMedia payload", {**payload, "monimaster-token": "***"})

    request = urllib.request.Request(
        LITMEDIA_API_URL,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers=build_litmedia_headers(payload["fingerprint"]),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LitMedia 接口返回 HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LitMedia 接口请求失败: {exc.reason}") from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw": response_text}


def call_litmedia_image_video(
    creative_params: CreativeParams,
    frontend_params: FrontendParams,
) -> dict[str, Any]:
    """调用 LitMedia 图生视频接口。"""

    payload = build_litmedia_i2v_payload(creative_params, frontend_params)
    debug_print("LitMedia i2v payload", {**payload, "monimaster-token": "***"})

    request = urllib.request.Request(
        LITMEDIA_IMAGE_VIDEO_API_URL,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers=build_litmedia_headers(payload["fingerprint"]),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LitMedia 图生接口返回 HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LitMedia 图生接口请求失败: {exc.reason}") from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw": response_text}


@tool(args_schema=SubmitVideoGenerationInput)
def submit_text_to_video(
    prompt: str,
    negative_prompt: str = "",
    video_model: str = "54",
    video_num: int = 1,
    ratio: str = "16:9",
    quality: str = "360p",
    duration: int = 5,
    open_filter: int = 0,
    sound_effect_switch: int = 1,
    seed: str = "",
    is_feed: int = 0,
) -> dict[str, Any]:
    """提交文生视频生成任务，返回 LitMedia 接口响应。"""

    creative_params = CreativeParams(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
    )
    frontend_params = FrontendParams(
        video_model=video_model,
        video_num=video_num,
        ratio=ratio,
        quality=quality,
        duration=duration,
        open_filter=open_filter,
        sound_effect_switch=sound_effect_switch,
        seed=seed,
        is_feed=is_feed,
    )
    return call_litmedia_text_video(creative_params, frontend_params)


@tool(args_schema=SubmitImageToVideoInput)
def submit_image_to_video(
    img_url: str,
    prompt: str,
    negative_prompt: str = "",
    video_model: str = "54",
    video_num: int = 1,
    quality: str = "360p",
    duration: int = 5,
    open_filter: int = 0,
    sound_effect_switch: int = 1,
    seed: str = "",
    is_feed: int = 0,
) -> dict[str, Any]:
    """提交图生视频生成任务，返回 LitMedia 接口响应。"""

    creative_params = CreativeParams(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        source_image_url=img_url,
    )
    frontend_params = FrontendParams(
        video_model=video_model,
        video_num=video_num,
        quality=quality,
        duration=duration,
        open_filter=open_filter,
        sound_effect_switch=sound_effect_switch,
        seed=seed,
        is_feed=is_feed,
    )
    return call_litmedia_image_video(creative_params, frontend_params)
