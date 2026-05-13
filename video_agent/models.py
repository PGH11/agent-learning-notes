"""Pydantic 协议与 LangGraph 状态定义。"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class CreativeParams(BaseModel):
    """Agent 负责维护的内容参数。"""

    prompt: str | None = Field(default=None, description="最终视频提示词")
    negative_prompt: str | None = Field(default=None, description="负向提示词")
    source_image_url: str | None = Field(default=None, description="图生视频的参考图 URL")


class FrontendParams(BaseModel):
    """前端页面选择的 UI 参数；CLI 里用默认值模拟。"""

    video_model: str = Field(default="54", description="前端选择的视频模型")
    video_num: int = Field(default=1, ge=1, le=4, description="前端选择的生成数量")
    ratio: str = Field(default="16:9", description="前端选择的视频比例")
    quality: str = Field(default="360p", description="前端选择的视频清晰度")
    duration: int = Field(default=5, ge=5, le=15, description="前端选择的视频时长")
    open_filter: int = Field(default=0, ge=0, le=1, description="是否开启过滤")
    sound_effect_switch: int = Field(default=1, ge=0, le=1, description="是否开启音效")
    seed: str = Field(default="", description="随机种子")
    is_feed: int = Field(default=0, ge=0, le=1, description="是否进入 feed")


class SubmitVideoGenerationInput(BaseModel):
    """提交文生视频生成任务的 tool 入参。"""

    prompt: str = Field(description="最终视频提示词")
    negative_prompt: str = Field(default="", description="负向提示词")
    video_model: str = Field(default="54", description="前端选择的视频模型")
    video_num: int = Field(default=1, ge=1, le=4, description="前端选择的生成数量")
    ratio: str = Field(default="16:9", description="前端选择的视频比例")
    quality: str = Field(default="360p", description="前端选择的视频清晰度")
    duration: int = Field(default=5, ge=5, le=15, description="前端选择的视频时长")
    open_filter: int = Field(default=0, ge=0, le=1, description="是否开启过滤")
    sound_effect_switch: int = Field(default=1, ge=0, le=1, description="是否开启音效")
    seed: str = Field(default="", description="随机种子")
    is_feed: int = Field(default=0, ge=0, le=1, description="是否进入 feed")


class SubmitImageToVideoInput(BaseModel):
    """提交图生视频生成任务的 tool 入参。"""

    img_url: str = Field(description="图生视频的参考图 URL")
    prompt: str = Field(description="图生视频提示词")
    negative_prompt: str = Field(default="", description="负向提示词")
    video_model: str = Field(default="54", description="前端选择的视频模型")
    video_num: int = Field(default=1, ge=1, le=4, description="前端选择的生成数量")
    quality: str = Field(default="360p", description="前端选择的视频清晰度")
    duration: int = Field(default=5, ge=5, le=15, description="前端选择的视频时长")
    open_filter: int = Field(default=0, ge=0, le=1, description="是否开启过滤")
    sound_effect_switch: int = Field(default=1, ge=0, le=1, description="是否开启音效")
    seed: str = Field(default="", description="随机种子")
    is_feed: int = Field(default=0, ge=0, le=1, description="是否进入 feed")


class CreativeParamsPatch(BaseModel):
    """LLM 本轮识别到的内容参数补丁。"""

    prompt: str | None = Field(default=None, description="本轮新增或修改的视频提示词")
    negative_prompt: str | None = Field(default=None, description="本轮新增或修改的负向提示词")
    source_image_url: str | None = Field(default=None, description="本轮新增或修改的参考图 URL")


class ToolCallRequest(BaseModel):
    """LLM 对工具调用的结构化决策。"""

    tool_name: Literal[
        "none",
        "submit_text_to_video",
        "submit_image_to_video",
    ] = Field(
        default="none",
        description="需要调用的工具名；不需要调用工具时返回 none",
    )
    reason: str = Field(default="", description="为什么需要或不需要调用工具")
    arguments: SubmitVideoGenerationInput | SubmitImageToVideoInput | None = Field(
        default=None,
        description="工具入参建议；最终执行前会由 Python 使用当前状态兜底校验",
    )


class CreativeUnderstanding(BaseModel):
    """LLM 对用户输入的结构化理解。"""

    intent: Literal[
        "chat",
        "brainstorm",
        "create_prompt",
        "update_prompt",
        "update_negative_prompt",
        "confirm",
        "cancel",
    ] = Field(description="用户当前意图")
    generation_mode: Literal["text_to_video", "image_to_video"] = Field(
        default="text_to_video",
        description="创作模式：文生视频或图生视频",
    )
    params_patch: CreativeParamsPatch = Field(
        default_factory=CreativeParamsPatch,
        description="需要合并到当前内容参数的补丁",
    )
    merge_strategy: Literal["fill_missing", "overwrite"] = Field(
        default="fill_missing",
        description="fill_missing 只填空字段；overwrite 表示用户明确要求修改已有内容",
    )
    reply: str = Field(description="要回复用户的自然语言内容")
    needs_clarification: bool = Field(default=False, description="是否需要继续追问")
    clarification_question: str | None = Field(default=None, description="需要追问的问题")
    ready_for_submit: bool = Field(default=False, description="内容参数是否已经足够提交")
    confirm_submit: bool = Field(default=False, description="用户是否明确确认提交")
    tool_call: ToolCallRequest = Field(
        default_factory=ToolCallRequest,
        description="用户确认提交后需要调用的工具",
    )


class ReflectionResult(BaseModel):
    """反思节点对创作结果和工具调用的审查结果。"""

    passed: bool = Field(description="本轮回复、prompt 和工具调用是否通过审查")
    should_block_tool: bool = Field(
        default=False,
        description="是否应该阻止本轮工具调用",
    )
    issues: list[str] = Field(default_factory=list, description="发现的问题")
    revised_reply: str | None = Field(
        default=None,
        description="如果需要修正给用户的回复，在这里返回修正版",
    )


class ChatTurn(BaseModel):
    """命令行会话历史。"""

    role: Literal["user", "assistant"]
    content: str


class RagDocument(BaseModel):
    """本地知识库检索到的片段。"""

    source: str
    content: str
    score: int


class MemoryRecord(BaseModel):
    """长期记忆文件里的统一条目结构。"""

    id: str
    category: Literal[
        "video_style",
        "avoid",
        "favorite_subject",
        "workflow_preference",
        "confirmed_prompt",
    ]
    value: str
    confidence: float = Field(ge=0, le=1)
    reason: str = ""
    source_text: str = ""
    created_at: str
    updated_at: str


class UserMemory(BaseModel):
    """长期记忆容器；外壳固定，条目内容可扩展。"""

    memories: list[MemoryRecord] = Field(default_factory=list)


class MemoryItem(BaseModel):
    """AI 判断后准备写入的一条长期记忆。"""

    category: Literal[
        "video_style",
        "avoid",
        "favorite_subject",
        "workflow_preference",
        "confirmed_prompt",
    ] = Field(description="记忆类别")
    value: str = Field(description="准备写入的记忆内容，必须简短、稳定、可复用")
    confidence: float = Field(
        ge=0,
        le=1,
        description="写入置信度；不确定时低于 0.75",
    )
    reason: str = Field(description="为什么这条信息值得长期保存")
    source_text: str = Field(description="支持这条记忆的用户原话或本轮上下文")


class MemoryWriteDecision(BaseModel):
    """AI 对本轮是否写入长期记忆的结构化判断。"""

    should_write: bool = Field(description="本轮是否有值得长期保存的信息")
    reason: str = Field(description="整体判断理由")
    memories: list[MemoryItem] = Field(default_factory=list, description="候选记忆")


class RetrievalDecision(BaseModel):
    """AI 对本轮是否需要 RAG 检索的结构化判断。"""

    should_retrieve: bool = Field(description="是否需要查询本地知识库")
    reason: str = Field(description="为什么需要或不需要检索")
    query: str | None = Field(
        default=None,
        description="需要检索时生成适合知识库搜索的查询；不需要时为 null",
    )


class CreativeState(TypedDict):
    """LangGraph 状态。"""

    user_input: str
    params: CreativeParams
    frontend_params: FrontendParams
    chat_history: list[ChatTurn]
    long_term_memory: str
    should_retrieve: bool
    retrieval_query: str
    rag_context: str
    rag_sources: list[str]
    route: Literal["chat", "creative", "i2v"]
    resolved_image_url: str
    reply: str
    is_ready: bool
    awaiting_confirmation: bool
    tool_call: ToolCallRequest
    reflection_passed: bool
    reflection_issues: list[str]


class RouteDecision(BaseModel):
    """路由节点的结构化输出。"""

    route: Literal["chat", "creative", "i2v"] = Field(description="chat、creative 或 i2v")
    reason: str = Field(description="路由原因")
