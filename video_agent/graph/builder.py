"""编译 LangGraph 状态机。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from video_agent.graph.nodes import (
    asset_resolver,
    chat_responder,
    creative_worker,
    i2v_worker,
    intent_router,
    memory_retriever,
    memory_writer,
    rag_retriever,
    reflection_worker,
    retrieval_router,
    tool_executor,
)
from video_agent.graph.routing import (
    route_after_reflection,
    route_after_retrieval,
    route_by_intent,
)
from video_agent.models import CreativeState


def build_graph():
    """构建视频创作助手状态机。

    一轮用户输入的大致路径（简化）::

        START
          → memory_retriever
          → retrieval_router
          → [rag_retriever | skip]
          → intent_router
          → chat_responder | creative_worker | asset_resolver → i2v_worker
          → …（创作经 reflection，可能 tool_executor）
          → memory_writer
          → END
    """

    graph = StateGraph(CreativeState)

    graph.add_node("memory_retriever", memory_retriever)
    graph.add_node("retrieval_router", retrieval_router)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("intent_router", intent_router)
    graph.add_node("chat_responder", chat_responder)
    graph.add_node("creative_worker", creative_worker)
    graph.add_node("asset_resolver", asset_resolver)
    graph.add_node("i2v_worker", i2v_worker)
    graph.add_node("reflection_worker", reflection_worker)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("memory_writer", memory_writer)

    graph.add_edge(START, "memory_retriever")
    graph.add_edge("memory_retriever", "retrieval_router")
    graph.add_conditional_edges(
        "retrieval_router",
        route_after_retrieval,
        {
            "retrieve": "rag_retriever",
            "skip": "intent_router",
        },
    )
    graph.add_edge("rag_retriever", "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "chat": "chat_responder",
            "creative": "creative_worker",
            "i2v": "asset_resolver",
        },
    )
    graph.add_edge("chat_responder", "memory_writer")
    graph.add_edge("creative_worker", "reflection_worker")
    graph.add_edge("asset_resolver", "i2v_worker")
    graph.add_edge("i2v_worker", "reflection_worker")
    graph.add_conditional_edges(
        "reflection_worker",
        route_after_reflection,
        {
            "tool": "tool_executor",
            "end": "memory_writer",
        },
    )
    graph.add_edge("tool_executor", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile()
