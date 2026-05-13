"""调试输出等横切能力。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from video_agent.settings import DEBUG_AI_OUTPUT


def debug_print(title: str, payload: Any) -> None:
    """打印 AI 调试信息；设置 DEBUG_AI_OUTPUT=0 可关闭。"""

    if not DEBUG_AI_OUTPUT:
        return

    if isinstance(payload, BaseModel):
        payload_text = payload.model_dump_json(ensure_ascii=False, indent=2)
    else:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

    print(f"\n[DEBUG] {title}:\n{payload_text}\n")
