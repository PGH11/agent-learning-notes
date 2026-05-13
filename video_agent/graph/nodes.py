"""LangGraph 各节点实现（纯函数 + 依赖注入在模块级 llm）。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from video_agent.graph.utils import history_to_text
from video_agent.litmedia_client import submit_image_to_video, submit_text_to_video
from video_agent.llm_factory import (
    creative_llm,
    memory_llm,
    reflection_llm,
    retrieval_llm,
    router_llm,
    llm,
)
from video_agent.logging_utils import debug_print
from video_agent.memory_store import (
    build_memory_record,
    format_user_memory,
    load_user_memory,
    memory_id_for,
    save_user_memory,
)
from video_agent.models import CreativeState, CreativeUnderstanding, ToolCallRequest
from video_agent.prompts import (
    CHAT_PROMPT,
    CREATIVE_PROMPT,
    I2V_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
    REFLECTION_PROMPT,
    RETRIEVAL_ROUTER_PROMPT,
    ROUTER_PROMPT,
)
from video_agent.rag_service import format_rag_context, search_knowledge_base


def extract_image_url(text: str) -> str:
    """从文本中提取第一个可用图片 URL。"""

    match = re.search(r"https?://[^\s]+?\.(?:png|jpg|jpeg|webp|gif)", text, re.I)
    return match.group(0) if match else ""


def ensure_creative_understanding(
    value: object,
    fallback_reply: str,
) -> CreativeUnderstanding:
    """把模型输出兜底为可用结构，避免返回 null 导致节点崩溃。"""

    if isinstance(value, CreativeUnderstanding):
        return value
    debug_print(
        "creative_understanding 兜底触发",
        {"raw_value": str(value)},
    )
    return CreativeUnderstanding(
        reply=fallback_reply,
        needs_clarification=True,
    )


def generate_clarification_reply(
    state: CreativeState,
    scenario: str,
) -> str:
    """当结构化输出异常时，用基础模型动态生成澄清回复（避免硬编码文案）。"""

    messages = [
        SystemMessage(
            content=(
                "你是视频创作助手。当前需要输出一条简短澄清回复，"
                "目标是让用户补齐继续执行所需信息。"
                "不要编造已执行结果，不要调用工具。"
            )
        ),
        HumanMessage(
            content=(
                f"场景：{scenario}\n\n"
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=8) or '暂无'}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip()


def invoke_structured_with_recovery(
    *,
    node_name: str,
    runnable: object,
    messages: list[SystemMessage | HumanMessage | AIMessage],
    state: CreativeState,
    schema_name: str,
):
    """结构化调用四层治理：主路径 -> 单次重试 -> 观测日志。"""

    def _observe(stage: str, payload: object) -> None:
        debug_print(
            "structured_output_observation",
            {
                "node": node_name,
                "schema": schema_name,
                "stage": stage,
                "is_null": payload is None,
                "user_input_preview": state["user_input"][:180],
                "awaiting_confirmation": state["awaiting_confirmation"],
                "has_prompt": bool(state["params"].prompt),
                "has_source_image": bool(state["params"].source_image_url),
            },
        )

    raw_output = runnable.invoke(messages)
    if raw_output is not None:
        return raw_output

    _observe("first_null", raw_output)
    retry_messages = [
        *messages,
        HumanMessage(
            content=(
                "恢复要求：你上一条结构化输出为空。"
                "请严格按既定 schema 返回一个合法对象，不要返回 null。"
            )
        ),
    ]
    retry_output = runnable.invoke(retry_messages)
    if retry_output is not None:
        _observe("retry_recovered", retry_output)
        return retry_output

    _observe("retry_still_null", retry_output)
    return None


def memory_retriever(state: CreativeState) -> CreativeState:
    """读取长期记忆，为本轮对话补充用户偏好。"""

    memory = load_user_memory()
    formatted_memory = format_user_memory(memory)
    debug_print("memory_retriever 长期记忆", {"memory": formatted_memory})
    return {
        **state,
        "long_term_memory": formatted_memory,
    }


def retrieval_router(state: CreativeState) -> CreativeState:
    """判断本轮是否需要查询知识库，并生成检索 query。"""

    messages = [
        SystemMessage(content=RETRIEVAL_ROUTER_PROMPT),
        HumanMessage(
            content=(
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "长期记忆：\n"
                f"{state['long_term_memory'] or '暂无长期记忆。'}\n\n"
                "是否正在等待最终确认：\n"
                f"{state['awaiting_confirmation']}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=6) or '暂无'}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    decision = retrieval_llm.invoke(messages)
    retrieval_query = (decision.query or "").strip()
    if not decision.should_retrieve or retrieval_query.lower() in {"null", "none"}:
        retrieval_query = ""
    debug_print(
        "retrieval_router AI 判断",
        {
            "should_retrieve": decision.should_retrieve,
            "reason": decision.reason,
            "retrieval_query": retrieval_query,
        },
    )

    return {
        **state,
        "should_retrieve": decision.should_retrieve,
        "retrieval_query": retrieval_query,
        "rag_context": "",
        "rag_sources": [],
    }


def asset_resolver(state: CreativeState) -> CreativeState:
    """解析图生所需的图片资产（当前版本先支持 URL）。"""

    candidate_url = (
        extract_image_url(state["user_input"])
        or state["params"].source_image_url
        or state["resolved_image_url"]
    )
    updated_params = state["params"].model_copy(deep=True)
    if candidate_url:
        updated_params.source_image_url = candidate_url

    debug_print(
        "asset_resolver 解析结果",
        {"resolved_image_url": candidate_url or None},
    )
    return {
        **state,
        "params": updated_params,
        "resolved_image_url": candidate_url,
    }


def memory_writer(state: CreativeState) -> CreativeState:
    """让 AI 判断本轮是否需要写入长期记忆，再由 Python 合并落盘。"""

    memory = load_user_memory()
    messages = [
        SystemMessage(content=MEMORY_EXTRACTOR_PROMPT),
        HumanMessage(
            content=(
                "当前长期记忆：\n"
                f"{format_user_memory(memory)}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}\n\n"
                "助手本轮回复：\n"
                f"{state['reply'] or '暂无'}\n\n"
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "是否已经完成提交或进入结束状态：\n"
                f"{state['is_ready']}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=6) or '暂无'}"
            )
        ),
    ]
    decision = memory_llm.invoke(messages)
    debug_print("memory_writer AI 判断", decision)

    accepted_memories = [
        item
        for item in decision.memories
        if decision.should_write and item.confidence >= 0.75 and item.value.strip()
    ]

    records_by_id = {record.id: record for record in memory.memories}
    for item in accepted_memories:
        value = item.value.strip()
        record_id = memory_id_for(item.category, value)
        existing_record = records_by_id.get(record_id)
        records_by_id[record_id] = build_memory_record(
            category=item.category,
            value=value,
            confidence=item.confidence,
            reason=item.reason,
            source_text=item.source_text,
            created_at=existing_record.created_at if existing_record else None,
        )

    memory.memories = list(records_by_id.values())
    confirmed_prompts = [
        record for record in memory.memories if record.category == "confirmed_prompt"
    ]
    if len(confirmed_prompts) > 5:
        keep_confirmed_ids = {record.id for record in confirmed_prompts[-5:]}
        memory.memories = [
            record
            for record in memory.memories
            if record.category != "confirmed_prompt" or record.id in keep_confirmed_ids
        ]

    save_user_memory(memory)
    debug_print(
        "memory_writer 写入长期记忆",
        {
            "accepted_memories": [item.model_dump() for item in accepted_memories],
            "memory": memory.model_dump(),
        },
    )
    return {
        **state,
        "long_term_memory": format_user_memory(memory),
    }


def rag_retriever(state: CreativeState) -> CreativeState:
    """检索本地知识库，为后续聊天或创作节点补充上下文。"""

    retrieval_query = state["retrieval_query"].strip() or state["user_input"]
    query = "\n".join(
        [
            retrieval_query,
            state["params"].model_dump_json(ensure_ascii=False, exclude_none=True),
            history_to_text(state["chat_history"], limit=4),
        ]
    )
    documents = search_knowledge_base(query)
    debug_print(
        "rag_retriever 检索结果",
        {
            "query": state["user_input"],
            "retrieval_query": retrieval_query,
            "sources": [doc.source for doc in documents],
            "scores": [doc.score for doc in documents],
        },
    )

    return {
        **state,
        "rag_context": format_rag_context(documents),
        "rag_sources": [doc.source for doc in documents],
    }


def intent_router(state: CreativeState) -> CreativeState:
    """判断当前输入是普通聊天还是创作任务。"""

    user_input_lower = state["user_input"].lower()
    has_image_url = bool(extract_image_url(state["user_input"]))
    has_i2v_intent = any(
        marker in user_input_lower
        for marker in [
            "图生",
            "图片生成视频",
            "根据这张图",
            "让这张图动",
            "image to video",
            "img2video",
            "i2v",
        ]
    )
    if has_image_url and has_i2v_intent:
        return {
            **state,
            "route": "i2v",
            "reply": "",
        }

    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(
            content=(
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "当前前端 UI 参数：\n"
                f"{state['frontend_params'].model_dump_json(ensure_ascii=False)}\n\n"
                "长期记忆：\n"
                f"{state['long_term_memory'] or '暂无长期记忆。'}\n\n"
                "知识库检索结果：\n"
                f"{state['rag_context'] or '暂无相关知识库资料。'}\n\n"
                f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=8) or '暂无'}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    decision = router_llm.invoke(messages)
    debug_print("intent_router AI 返回", decision)

    return {
        **state,
        "route": decision.route,
        "reply": "",
    }


def chat_responder(state: CreativeState) -> CreativeState:
    """处理闲聊、能力介绍和上下文问答。"""

    messages = [
        SystemMessage(
            content=(
                f"{CHAT_PROMPT}\n\n"
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "当前前端 UI 参数：\n"
                f"{state['frontend_params'].model_dump_json(ensure_ascii=False)}\n\n"
                "长期记忆：\n"
                f"{state['long_term_memory'] or '暂无长期记忆。'}\n\n"
                "知识库检索结果：\n"
                f"{state['rag_context'] or '暂无相关知识库资料。'}"
            )
        )
    ]
    for turn in state["chat_history"][-10:]:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=state["user_input"]))

    response = llm.invoke(messages)
    debug_print("chat_responder AI 返回", {"content": str(response.content)})

    return {
        **state,
        "reply": str(response.content),
        "is_ready": False,
    }


def creative_worker(state: CreativeState) -> CreativeState:
    """理解用户创意，更新 prompt，并生成回复。"""

    current_params = state["params"]
    messages = [
        SystemMessage(content=CREATIVE_PROMPT),
        HumanMessage(
            content=(
                "当前内容参数：\n"
                f"{current_params.model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "当前前端 UI 参数：\n"
                f"{state['frontend_params'].model_dump_json(ensure_ascii=False)}\n\n"
                "长期记忆：\n"
                f"{state['long_term_memory'] or '暂无长期记忆。'}\n\n"
                "知识库检索结果：\n"
                f"{state['rag_context'] or '暂无相关知识库资料。'}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=10) or '暂无'}\n\n"
                f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    raw_understanding = invoke_structured_with_recovery(
        node_name="creative_worker",
        runnable=creative_llm,
        messages=messages,
        state=state,
        schema_name="CreativeUnderstanding",
    )
    fallback_reply = generate_clarification_reply(
        state,
        scenario="creative_worker 结构化输出为空，需要继续澄清创作需求。",
    )
    understanding = ensure_creative_understanding(raw_understanding, fallback_reply)
    debug_print("creative_worker AI 返回", understanding)

    updated_params = current_params.model_copy(deep=True)
    patch = understanding.params_patch
    allow_overwrite = understanding.merge_strategy == "overwrite"

    if patch.prompt and (updated_params.prompt is None or allow_overwrite):
        updated_params.prompt = patch.prompt.strip()
    if patch.negative_prompt and (
        updated_params.negative_prompt is None or allow_overwrite
    ):
        updated_params.negative_prompt = patch.negative_prompt.strip()
    if patch.source_image_url and (
        updated_params.source_image_url is None or allow_overwrite
    ):
        updated_params.source_image_url = patch.source_image_url.strip()

    is_ready = bool(
        state["awaiting_confirmation"]
        and understanding.confirm_submit
        and updated_params.prompt
    )
    awaiting_confirmation = bool(
        understanding.ready_for_submit
        and updated_params.prompt
        and not is_ready
    )

    return {
        **state,
        "params": updated_params,
        "reply": understanding.reply,
        "is_ready": False if understanding.tool_call.tool_name != "none" else is_ready,
        "awaiting_confirmation": awaiting_confirmation,
        "tool_call": understanding.tool_call,
    }


def i2v_worker(state: CreativeState) -> CreativeState:
    """图生视频专用创作节点。"""

    current_params = state["params"].model_copy(deep=True)
    if state["resolved_image_url"] and not current_params.source_image_url:
        current_params.source_image_url = state["resolved_image_url"]

    messages = [
        SystemMessage(content=I2V_PROMPT),
        HumanMessage(
            content=(
                "当前内容参数：\n"
                f"{current_params.model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "当前前端 UI 参数：\n"
                f"{state['frontend_params'].model_dump_json(ensure_ascii=False)}\n\n"
                "长期记忆：\n"
                f"{state['long_term_memory'] or '暂无长期记忆。'}\n\n"
                "知识库检索结果：\n"
                f"{state['rag_context'] or '暂无相关知识库资料。'}\n\n"
                "最近对话：\n"
                f"{history_to_text(state['chat_history'], limit=10) or '暂无'}\n\n"
                "解析到的图片 URL：\n"
                f"{state['resolved_image_url'] or '暂无'}\n\n"
                f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    raw_understanding = invoke_structured_with_recovery(
        node_name="i2v_worker",
        runnable=creative_llm,
        messages=messages,
        state=state,
        schema_name="CreativeUnderstanding",
    )
    fallback_reply = generate_clarification_reply(
        state,
        scenario="i2v_worker 结构化输出为空，需要澄清图生视频所需信息（包含图片 URL 与动作诉求）。",
    )
    understanding = ensure_creative_understanding(raw_understanding, fallback_reply)
    debug_print("i2v_worker AI 返回", understanding)

    updated_params = current_params.model_copy(deep=True)
    patch = understanding.params_patch
    allow_overwrite = understanding.merge_strategy == "overwrite"

    if patch.prompt and (updated_params.prompt is None or allow_overwrite):
        updated_params.prompt = patch.prompt.strip()
    if patch.negative_prompt and (
        updated_params.negative_prompt is None or allow_overwrite
    ):
        updated_params.negative_prompt = patch.negative_prompt.strip()
    if patch.source_image_url and (
        updated_params.source_image_url is None or allow_overwrite
    ):
        updated_params.source_image_url = patch.source_image_url.strip()

    if state["resolved_image_url"] and not updated_params.source_image_url:
        updated_params.source_image_url = state["resolved_image_url"]

    has_asset = bool(updated_params.source_image_url)
    is_ready = bool(
        state["awaiting_confirmation"]
        and understanding.confirm_submit
        and updated_params.prompt
        and has_asset
    )
    awaiting_confirmation = bool(
        understanding.ready_for_submit
        and updated_params.prompt
        and has_asset
        and not is_ready
    )

    if understanding.tool_call.tool_name == "submit_text_to_video":
        understanding.tool_call.tool_name = "submit_image_to_video"

    if not has_asset:
        understanding.tool_call = ToolCallRequest(
            tool_name="none",
            reason="图生模式缺少图片 URL，先要求用户提供图片。",
        )
        awaiting_confirmation = False
        is_ready = False

    return {
        **state,
        "params": updated_params,
        "reply": understanding.reply,
        "is_ready": False if understanding.tool_call.tool_name != "none" else is_ready,
        "awaiting_confirmation": awaiting_confirmation,
        "tool_call": understanding.tool_call,
    }


def reflection_worker(state: CreativeState) -> CreativeState:
    """反思审查创作结果，必要时阻止工具调用。"""

    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(
            content=(
                "当前内容参数：\n"
                f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
                "当前回复：\n"
                f"{state['reply']}\n\n"
                "是否正在等待最终确认：\n"
                f"{state['awaiting_confirmation']}\n\n"
                "工具调用请求：\n"
                f"{state['tool_call'].model_dump_json(ensure_ascii=False)}\n\n"
                "图生图片 URL：\n"
                f"{state['params'].source_image_url or '暂无'}\n\n"
                "用户最新输入：\n"
                f"{state['user_input']}"
            )
        ),
    ]
    result = reflection_llm.invoke(messages)
    debug_print("reflection_worker AI 返回", result)

    if (
        state["tool_call"].tool_name == "submit_image_to_video"
        and not state["params"].source_image_url
    ):
        result.should_block_tool = True
        result.passed = False
        result.issues = [*result.issues, "图生工具调用缺少 source_image_url。"]
        if not result.revised_reply:
            result.revised_reply = generate_clarification_reply(
                state,
                scenario="reflection_worker 拦截图生提交，原因是缺少可访问图片 URL。",
            )

    blocked_tool_call = ToolCallRequest() if result.should_block_tool else state["tool_call"]
    return {
        **state,
        "reply": result.revised_reply or state["reply"],
        "tool_call": blocked_tool_call,
        "awaiting_confirmation": (
            False if result.should_block_tool else state["awaiting_confirmation"]
        ),
        "reflection_passed": result.passed,
        "reflection_issues": result.issues,
    }


def tool_executor(state: CreativeState) -> CreativeState:
    """执行 Agent 选择的工具；所有工具调用都通过白名单控制。"""

    tool_call = state["tool_call"]
    if tool_call.tool_name not in {"submit_text_to_video", "submit_image_to_video"}:
        return {
            **state,
            "reply": state["reply"],
            "is_ready": False,
        }

    frontend_params = state["frontend_params"]
    tool_args = {**frontend_params.model_dump()}
    if tool_call.tool_name == "submit_text_to_video":
        tool_args.update(
            {
                "prompt": state["params"].prompt or "",
                "negative_prompt": state["params"].negative_prompt or "",
            }
        )
    else:
        tool_args.update(
            {
                "img_url": state["params"].source_image_url or "",
                "prompt": state["params"].prompt or "",
                "negative_prompt": state["params"].negative_prompt or "",
            }
        )
    debug_print(
        "tool_executor 执行工具",
        {
            "tool_name": tool_call.tool_name,
            "reason": tool_call.reason,
            "tool_args": tool_args,
        },
    )

    try:
        if tool_call.tool_name == "submit_text_to_video":
            result = submit_text_to_video.invoke(tool_args)
        else:
            result = submit_image_to_video.invoke(tool_args)
    except Exception as exc:
        return {
            **state,
            "reply": f"{state['reply']}\n工具 {tool_call.tool_name} 执行失败：{exc}",
            "is_ready": True,
            "awaiting_confirmation": False,
            "tool_call": ToolCallRequest(),
        }

    return {
        **state,
        "reply": (
            f"{state['reply']}\n"
            f"工具 {tool_call.tool_name} 已执行，接口响应：\n"
            f"{json.dumps(result, ensure_ascii=False, indent=2)}"
        ),
        "is_ready": True,
        "awaiting_confirmation": False,
        "tool_call": ToolCallRequest(),
    }
