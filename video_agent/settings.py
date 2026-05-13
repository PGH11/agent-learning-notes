"""项目级配置与路径（相对仓库根目录）。"""

from __future__ import annotations

import os
from pathlib import Path

# `video_agent/settings.py` → 仓库根
LANG_ROOT: Path = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR: Path = LANG_ROOT / "knowledge"
MEMORY_FILE: Path = LANG_ROOT / "memory" / "user_memory.json"

LITMEDIA_API_URL = "https://litvideo-api.litmedia.ai/lit-video/do-text-video"

DEBUG_AI_OUTPUT = os.getenv("DEBUG_AI_OUTPUT", "1") != "0"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_CHUNK_MAX_CHARS = int(os.getenv("RAG_CHUNK_MAX_CHARS", "900"))

# LangSmith：仅当设置了 LANGSMITH_API_KEY 时启用（勿在代码中写死密钥）
LANGSMITH_API_KEY_VALUE = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_PROJECT_VALUE = os.getenv("LANGSMITH_PROJECT", "").strip()
LANGSMITH_ENDPOINT_VALUE = os.getenv(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com",
).strip()


def configure_langsmith() -> None:
    """配置 LangSmith tracing。"""

    if not LANGSMITH_API_KEY_VALUE:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT_VALUE
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY_VALUE
    if LANGSMITH_PROJECT_VALUE:
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT_VALUE


configure_langsmith()
