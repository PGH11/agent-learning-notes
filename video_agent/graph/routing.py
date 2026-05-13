"""LangGraph 条件边路由函数。"""

from __future__ import annotations

from typing import Literal

from video_agent.models import CreativeState


def route_after_retrieval(state: CreativeState) -> Literal["retrieve", "skip"]:
    """根据 retrieval_router 的判断决定是否进入 RAG 节点。"""

    if state["should_retrieve"] and state["retrieval_query"].strip():
        return "retrieve"
    return "skip"


def route_by_intent(state: CreativeState) -> Literal["chat", "creative", "i2v"]:
    """映射 LangGraph 分支。"""

    return state["route"]


def route_after_reflection(state: CreativeState) -> Literal["tool", "end"]:
    """反思通过后，根据工具调用决策进入工具节点或结束。"""

    if (
        state["tool_call"].tool_name == "submit_text_to_video"
        and state["params"].prompt
    ):
        return "tool"
    if (
        state["tool_call"].tool_name == "submit_image_to_video"
        and state["params"].prompt
        and state["params"].source_image_url
    ):
        return "tool"
    return "end"
