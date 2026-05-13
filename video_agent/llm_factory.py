"""LLM 客户端与结构化输出 Runnable 工厂。"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from video_agent.models import (
    CreativeUnderstanding,
    MemoryWriteDecision,
    ReflectionResult,
    RetrievalDecision,
    RouteDecision,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def build_llm() -> ChatOpenAI:
    """创建聊天模型。"""

    if load_dotenv is not None:
        load_dotenv()

    ark_api_key = (os.getenv("ARK_API_KEY") or "").strip()
    if not ark_api_key:
        raise ValueError("未检测到 ARK_API_KEY，请先配置环境变量后再运行。")

    return ChatOpenAI(
        api_key=ark_api_key,
        base_url=os.getenv(
            "ARK_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/coding/v3",
        ),
        model=os.getenv("ARK_MODEL", "ark-code-latest"),
        temperature=0,
    )


llm = build_llm()
router_llm = llm.with_structured_output(RouteDecision, method="function_calling")
creative_llm = llm.with_structured_output(CreativeUnderstanding, method="function_calling")
reflection_llm = llm.with_structured_output(ReflectionResult, method="function_calling")
memory_llm = llm.with_structured_output(MemoryWriteDecision, method="function_calling")
retrieval_llm = llm.with_structured_output(RetrievalDecision, method="function_calling")
