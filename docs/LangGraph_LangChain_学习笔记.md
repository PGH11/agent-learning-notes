# LangGraph 与 LangChain 学习笔记

这份文档结合当前目录里的 `image_agent.py`，解释脚本里 LangChain 和 LangGraph 分别做什么，以及一次用户对话是怎么被送进 AI、怎么更新状态、怎么最终调用接口的。

## 一句话理解

在这个脚本里：

- `LangChain` 负责和大模型交互：创建模型客户端、组织消息、调用 AI、拿结构化输出。
- `LangGraph` 负责流程编排：定义状态、节点、分支，让一轮对话按固定流程运行。
- `Pydantic` 负责数据结构：约束 AI 必须返回什么字段，避免只返回一段不可控文本。
- 普通 Python 负责业务逻辑：合并参数、生成签名、调用 LitMedia 接口。

可以理解成：

```text
LangChain = 模型调用层
LangGraph = 状态机编排层
Pydantic = 数据协议层
Python = 业务执行层
```

## 当前脚本的整体流程

用户每输入一句话，脚本会跑一次 LangGraph：

```text
User 输入
  ↓
intent_router
  ↓
根据 route 分支
  ├─ chat_responder
  └─ creative_worker
  ↓
END
```

如果用户是在闲聊，比如：

```text
你能帮我干嘛？
```

会走：

```text
intent_router -> chat_responder -> END
```

如果用户是在创作视频，比如：

```text
我想生成一个关于小狗的视频
```

会走：

```text
intent_router -> creative_worker -> END
```

当用户最终确认后，CLI 会在 LangGraph 之外调用：

```text
call_litmedia_text_video()
```

## 1. LangChain 在脚本里的作用

### 1.1 创建模型客户端

脚本里通过 `ChatOpenAI` 创建模型：

```python
llm = ChatOpenAI(
    api_key=ark_api_key,
    base_url=os.getenv(
        "ARK_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/coding/v3",
    ),
    model=os.getenv("ARK_MODEL", "ark-code-latest"),
    temperature=0,
)
```

虽然类名叫 `ChatOpenAI`，但这里配置了 Ark 的 `base_url`，所以实际是通过 OpenAI 兼容协议调用 Ark 模型。

这里的关键参数：

- `api_key`：模型调用密钥。
- `base_url`：模型接口地址。
- `model`：模型名称。
- `temperature=0`：降低随机性，让结构化任务更稳定。

### 1.2 组织消息上下文

LangChain 使用消息对象和模型对话：

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
```

三类消息的作用：

- `SystemMessage`：告诉模型角色、规则、输出格式。
- `HumanMessage`：用户输入，或程序拼出来的当前上下文。
- `AIMessage`：历史助手回复。

例如 `creative_worker()` 里会构造：

```python
messages = [
    SystemMessage(content=CREATIVE_PROMPT),
    HumanMessage(
        content=(
            "当前内容参数：\n"
            f"{current_params.model_dump_json(...)}\n\n"
            "最近对话：\n"
            f"{history_to_text(...)}\n\n"
            f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
            "用户最新输入：\n"
            f"{state['user_input']}"
        )
    ),
]
```

这就是实际发给 AI 的上下文。

### 1.3 普通文本输出

`chat_responder()` 用普通 `llm.invoke(messages)`：

```python
response = llm.invoke(messages)
```

这种调用让 AI 自由返回一段文本，适合：

- 闲聊
- 能力介绍
- 回答“你记得我刚刚说了什么吗”

返回后脚本用：

```python
str(response.content)
```

拿到回复文本。

### 1.4 结构化输出

创作节点不是让 AI 随便返回文本，而是要求 AI 返回固定结构。

脚本里有：

```python
creative_llm = llm.with_structured_output(CreativeUnderstanding)
```

`CreativeUnderstanding` 是 Pydantic 模型：

```python
class CreativeUnderstanding(BaseModel):
    intent: Literal[
        "chat",
        "brainstorm",
        "create_prompt",
        "update_prompt",
        "update_negative_prompt",
        "confirm",
        "cancel",
    ]
    params_patch: CreativeParamsPatch
    merge_strategy: Literal["fill_missing", "overwrite"]
    reply: str
    needs_clarification: bool
    clarification_question: str | None
    ready_for_submit: bool
    confirm_submit: bool
```

这表示 AI 必须返回类似这样的结构：

```json
{
  "intent": "create_prompt",
  "params_patch": {
    "prompt": "短腿柯基在洒满午后阳光的客厅里追彩色毛绒球...",
    "negative_prompt": "画面昏暗、镜头剧烈晃动、场景杂乱"
  },
  "merge_strategy": "fill_missing",
  "reply": "我已经帮你整理成完整视频提示词啦。",
  "needs_clarification": false,
  "clarification_question": null,
  "ready_for_submit": true,
  "confirm_submit": false
}
```

这样代码可以稳定读取：

```python
understanding.params_patch.prompt
understanding.ready_for_submit
understanding.confirm_submit
```

这就是 `with_structured_output` 的价值。

## 2. Pydantic 在脚本里的作用

Pydantic 用来定义“数据协议”。

### 2.1 Agent 内容参数

```python
class CreativeParams(BaseModel):
    prompt: str | None = Field(default=None, description="最终视频提示词")
    negative_prompt: str | None = Field(default=None, description="负向提示词")
```

这是 Agent 真正负责维护的内容。

注意：比例、时长、模型、清晰度这些已经不属于 Agent 内容参数，而是交给前端。

### 2.2 前端 UI 参数

```python
class FrontendParams(BaseModel):
    video_model: str = Field(default="54")
    video_num: int = Field(default=1, ge=1, le=4)
    ratio: str = Field(default="16:9")
    quality: str = Field(default="360p")
    duration: int = Field(default=5, ge=5, le=15)
```

CLI 没有前端页面，所以用 `FrontendParams()` 模拟前端已经选择好的值。

未来接真实前端时，可以把页面选中的参数传进来，替换这个默认对象。

### 2.3 参数补丁

```python
class CreativeParamsPatch(BaseModel):
    prompt: str | None = None
    negative_prompt: str | None = None
```

每轮 AI 不直接改完整状态，而是返回一个“补丁”。

例如用户说：

```text
改成更搞笑一点
```

AI 可能返回：

```json
{
  "params_patch": {
    "prompt": "一只短腿柯基在客厅里追球，结果滑进抱枕堆..."
  },
  "merge_strategy": "overwrite"
}
```

然后 Python 决定怎么把这个补丁合并进当前状态。

## 3. LangGraph 在脚本里的作用

LangGraph 负责把多个节点组织成状态机。

### 3.1 状态定义

```python
class CreativeState(TypedDict):
    user_input: str
    params: CreativeParams
    chat_history: list[ChatTurn]
    route: Literal["chat", "creative"]
    reply: str
    is_ready: bool
    awaiting_confirmation: bool
```

这就是每一轮图执行时传来传去的状态。

可以理解为一个字典：

```json
{
  "user_input": "我想生成一个关于小狗的视频",
  "params": {
    "prompt": null,
    "negative_prompt": null
  },
  "chat_history": [],
  "route": "chat",
  "reply": "",
  "is_ready": false,
  "awaiting_confirmation": false
}
```

每个节点都会读取这个状态，并返回修改后的状态。

### 3.2 创建图

```python
graph = StateGraph(CreativeState)
```

这表示：我要创建一个状态图，状态结构是 `CreativeState`。

### 3.3 添加节点

```python
graph.add_node("intent_router", intent_router)
graph.add_node("chat_responder", chat_responder)
graph.add_node("creative_worker", creative_worker)
```

每个节点都是一个普通 Python 函数。

节点函数的形式是：

```python
def node_name(state: CreativeState) -> CreativeState:
    ...
    return new_state
```

### 3.4 添加边

```python
graph.add_edge(START, "intent_router")
```

表示每轮开始先执行 `intent_router`。

### 3.5 条件分支

```python
graph.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "chat": "chat_responder",
        "creative": "creative_worker",
    },
)
```

含义是：

1. 执行完 `intent_router`。
2. 调用 `route_by_intent(state)`。
3. 如果返回 `"chat"`，走 `chat_responder`。
4. 如果返回 `"creative"`，走 `creative_worker`。

`route_by_intent` 很简单：

```python
def route_by_intent(state: CreativeState) -> Literal["chat", "creative"]:
    return state["route"]
```

而 `state["route"]` 是 `intent_router` 里由 AI 判断出来的。

### 3.6 结束节点

```python
graph.add_edge("chat_responder", END)
graph.add_edge("creative_worker", END)
```

表示两个分支执行完就结束本轮图。

### 3.7 编译图

```python
return graph.compile()
```

编译后得到可调用的 `app`：

```python
app = build_graph()
state = app.invoke(state)
```

这时 LangGraph 会按照你定义的节点和边自动执行。

## 4. 三个核心节点

### 4.1 `intent_router`

职责：判断这句话走聊天，还是走创作。

输入：

```text
用户最新输入
当前内容参数
最近对话
是否等待确认
```

调用：

```python
decision = router_llm.invoke(messages)
```

输出：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

然后写回状态：

```python
return {
    **state,
    "route": decision.route,
    "reply": "",
}
```

### 4.2 `chat_responder`

职责：处理闲聊、能力介绍、上下文问答。

例如：

```text
你能帮我干嘛？
```

这里使用普通文本模型调用：

```python
response = llm.invoke(messages)
```

然后把文本放进：

```python
"reply": str(response.content)
```

### 4.3 `creative_worker`

职责：处理创作任务。

例如：

```text
我想生成一个关于小狗的视频
```

它会调用：

```python
understanding = creative_llm.invoke(messages)
```

因为 `creative_llm` 使用了：

```python
with_structured_output(CreativeUnderstanding)
```

所以返回的是结构化对象，而不是普通字符串。

然后代码合并参数：

```python
updated_params = current_params.model_copy(deep=True)
patch = understanding.params_patch
allow_overwrite = understanding.merge_strategy == "overwrite"

if patch.prompt and (updated_params.prompt is None or allow_overwrite):
    updated_params.prompt = patch.prompt.strip()
```

如果用户明确确认：

```python
is_ready = bool(
    state["awaiting_confirmation"]
    and understanding.confirm_submit
    and updated_params.prompt
)
```

如果内容已经可提交但还没确认：

```python
awaiting_confirmation = bool(
    understanding.ready_for_submit
    and updated_params.prompt
    and not is_ready
)
```

## 5. 一次完整对话是怎么跑的

以这段为例：

```text
User: 我想生成一个关于小狗的视频
```

### 第一步：CLI 写入状态

```python
state = {
    **state,
    "user_input": user_input,
    "reply": "",
    "is_ready": False,
}
```

### 第二步：调用 LangGraph

```python
state = app.invoke(state)
```

### 第三步：进入 `intent_router`

AI 判断：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

### 第四步：进入 `creative_worker`

AI 返回：

```json
{
  "intent": "brainstorm",
  "params_patch": {
    "prompt": null,
    "negative_prompt": null
  },
  "reply": "我为你准备了3个不同风格的小狗视频创意方向...",
  "needs_clarification": true,
  "ready_for_submit": false,
  "confirm_submit": false
}
```

### 第五步：CLI 打印回复并保存历史

```python
print(f"Agent: {state['reply']}")
state["chat_history"].extend(
    [
        ChatTurn(role="user", content=user_input),
        ChatTurn(role="assistant", content=state["reply"]),
    ]
)
```

## 6. 最终确认与接口调用

当用户输入：

```text
确认生成
```

如果当前已经处于 `awaiting_confirmation=True`，并且 AI 返回：

```json
{
  "confirm_submit": true
}
```

则：

```python
is_ready = True
```

循环结束：

```python
while not state["is_ready"]:
    ...
```

然后执行：

```python
result = call_litmedia_text_video(state["params"], frontend_params)
```

这里的 `state["params"]` 是 Agent 生成的内容参数：

```json
{
  "prompt": "...",
  "negative_prompt": "..."
}
```

`frontend_params` 是前端 UI 参数：

```json
{
  "video_model": "54",
  "video_num": 1,
  "ratio": "16:9",
  "quality": "360p",
  "duration": 5
}
```

两者合并成接口 payload。

## 7. Agent 参数与前端参数如何避免冲突

当前设计里：

```text
Agent 负责 prompt / negative_prompt
前端负责 ratio / duration / model / quality / video_num
```

所以不会出现：

```text
Agent 说 9:16
前端选 16:9
```

这种冲突。

如果未来希望用户在聊天里说：

```text
我要竖屏，5 秒
```

建议不要让 Agent 直接覆盖前端参数，而是返回：

```json
{
  "ui_params_suggestion": {
    "ratio": "9:16",
    "duration": 5
  }
}
```

然后前端提示用户：

```text
检测到你想改为竖屏 5 秒，是否应用到右侧参数？
```

最终仍以前端控件为准。

## 8. 调试输出怎么看

当前脚本默认关闭调试输出：

```python
DEBUG_AI_OUTPUT = os.getenv("DEBUG_AI_OUTPUT", "0") != "0"
```

含义是：

- 如果没有设置环境变量，`os.getenv("DEBUG_AI_OUTPUT", "0")` 会得到 `"0"`，所以调试关闭。
- 如果环境变量不是 `"0"`，调试开启。

临时开启方式：

```powershell
$env:DEBUG_AI_OUTPUT="1"
python image_agent.py
```

开启后终端会看到：

```text
[DEBUG] intent_router AI 返回
[DEBUG] creative_worker AI 返回
[DEBUG] LitMedia payload
```

关闭方式：

```powershell
$env:DEBUG_AI_OUTPUT="0"
python image_agent.py
```

如果想看完整发给 AI 的上下文，可以继续扩展一个函数：

```python
def debug_messages(title: str, messages: list) -> None:
    if not DEBUG_AI_OUTPUT:
        return
    print(f"\n[DEBUG] {title}")
    for index, message in enumerate(messages, start=1):
        print(f"\n--- message {index}: {message.type} ---")
        print(message.content)
```

然后在 `router_llm.invoke(messages)` 或 `creative_llm.invoke(messages)` 前打印。

## 9. 当前代码里 AI 到底看到了什么上下文

你的一整个流程里，真正给 AI 看上下文的地方有三处。

### 9.1 `intent_router()` 给路由 AI 的上下文

代码位置：

```python
messages = [
    SystemMessage(content=ROUTER_PROMPT),
    HumanMessage(
        content=(
            "当前内容参数：\n"
            f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
            f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
            "最近对话：\n"
            f"{history_to_text(state['chat_history'], limit=8) or '暂无'}\n\n"
            "用户最新输入：\n"
            f"{state['user_input']}"
        )
    ),
]
```

AI 看到的信息包括：

- `ROUTER_PROMPT`：路由规则。
- 当前内容参数：已有的 `prompt / negative_prompt`。
- 是否正在等待确认：`awaiting_confirmation`。
- 最近 8 条对话：`history_to_text(..., limit=8)`。
- 用户最新输入。

这个节点只负责判断：

```json
{
  "route": "chat 或 creative",
  "reason": "路由原因"
}
```

### 9.2 `chat_responder()` 给聊天 AI 的上下文

代码位置：

```python
messages = [
    SystemMessage(
        content=(
            f"{CHAT_PROMPT}\n\n"
            "当前内容参数：\n"
            f"{state['params'].model_dump_json(ensure_ascii=False, exclude_none=True)}"
        )
    )
]
for turn in state["chat_history"][-10:]:
    if turn.role == "user":
        messages.append(HumanMessage(content=turn.content))
    else:
        messages.append(AIMessage(content=turn.content))
messages.append(HumanMessage(content=state["user_input"]))
```

AI 看到的信息包括：

- `CHAT_PROMPT`：告诉它自己是视频创作助手。
- 当前内容参数。
- 最近 10 条原始对话，且按 `HumanMessage / AIMessage` 类型传入。
- 用户最新输入。

这个节点最接近普通聊天，所以适合回答：

```text
你能帮我干嘛？
你记得我刚刚说了什么吗？
我现在的 prompt 是什么？
```

### 9.3 `creative_worker()` 给创作 AI 的上下文

代码位置：

```python
messages = [
    SystemMessage(content=CREATIVE_PROMPT),
    HumanMessage(
        content=(
            "当前内容参数：\n"
            f"{current_params.model_dump_json(ensure_ascii=False, exclude_none=True)}\n\n"
            "最近对话：\n"
            f"{history_to_text(state['chat_history'], limit=10) or '暂无'}\n\n"
            f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
            "用户最新输入：\n"
            f"{state['user_input']}"
        )
    ),
]
```

AI 看到的信息包括：

- `CREATIVE_PROMPT`：创作规则、产品边界、结构化输出要求。
- 当前内容参数。
- 最近 10 条对话文本。
- 是否正在等待确认。
- 用户最新输入。

这个节点负责生成结构化结果：

```json
{
  "intent": "brainstorm/create_prompt/update_prompt/confirm",
  "params_patch": {
    "prompt": "...",
    "negative_prompt": "..."
  },
  "merge_strategy": "fill_missing/overwrite",
  "reply": "...",
  "ready_for_submit": true,
  "confirm_submit": false
}
```

## 10. `limit=8/10` 是什么

这不是模型限制，而是代码里人为决定“给 AI 看最近多少条对话”。

例如：

```python
history_to_text(state["chat_history"], limit=10)
```

内部实际做的是：

```python
history[-10:]
```

也就是只取最近 10 条历史。

为什么不全部给 AI？

- 上下文越长，token 消耗越高。
- 模型响应更慢。
- 很久以前的信息可能干扰当前任务。
- 超过模型上下文窗口后会被截断或报错。

企业版常见做法是：

```text
长期摘要 memory_summary + 最近 10-20 条原文
```

当前脚本还没有 `memory_summary`，只有：

```python
chat_history: list[ChatTurn]
```

所以现在 AI 主要依靠：

```text
当前 params + 最近若干条 chat_history
```

来理解上下文。

## 11. `RouteDecision` 那段动态模型是什么意思

当前代码里有这一段：

```python
router_llm = llm.with_structured_output(
    type(
        "RouteDecision",
        (BaseModel,),
        {
            "__annotations__": {
                "route": Literal["chat", "creative"],
                "reason": str,
            },
            "route": Field(description="chat 或 creative"),
            "reason": Field(description="路由原因"),
        },
    )
)
```

这段是在用 Python 的 `type()` 动态创建一个 Pydantic 类。

它等价于手写：

```python
class RouteDecision(BaseModel):
    route: Literal["chat", "creative"] = Field(description="chat 或 creative")
    reason: str = Field(description="路由原因")

router_llm = llm.with_structured_output(RouteDecision)
```

作用是要求路由 AI 返回固定结构：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

从可读性上看，建议以后改成显式类：

```python
class RouteDecision(BaseModel):
    route: Literal["chat", "creative"] = Field(description="chat 或 creative")
    reason: str = Field(description="路由原因")
```

这样比动态 `type(...)` 更适合学习和维护。

## 12. 如何给不同路由用不同模型

当前代码里所有节点都共用同一个基础模型：

```python
llm = build_llm()
router_llm = llm.with_structured_output(...)
creative_llm = llm.with_structured_output(CreativeUnderstanding)
```

如果想不同节点用不同模型，可以把 `build_llm()` 改成支持传模型名：

```python
def build_llm(model: str | None = None) -> ChatOpenAI:
    ark_api_key = os.getenv("ARK_API_KEY") or DEFAULT_ARK_API_KEY

    return ChatOpenAI(
        api_key=ark_api_key,
        base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"),
        model=model or os.getenv("ARK_MODEL", "ark-code-latest"),
        temperature=0,
    )
```

然后分别创建：

```python
router_base_llm = build_llm(os.getenv("ARK_ROUTER_MODEL", "ark-code-latest"))
chat_llm = build_llm(os.getenv("ARK_CHAT_MODEL", "ark-code-latest"))
creative_base_llm = build_llm(os.getenv("ARK_CREATIVE_MODEL", "ark-code-latest"))

router_llm = router_base_llm.with_structured_output(RouteDecision)
creative_llm = creative_base_llm.with_structured_output(CreativeUnderstanding)
```

对应关系可以是：

```text
intent_router：快模型、便宜模型
chat_responder：普通对话模型
creative_worker：创作能力更强的模型
```

## 13. 最终接口调用不在 LangGraph 里

当前设计里，LangGraph 只负责对话和内容确认。

真正调用接口发生在 `run_cli()` 循环结束后：

```python
while not state["is_ready"]:
    ...

result = call_litmedia_text_video(state["params"], frontend_params)
```

也就是说：

```text
LangGraph：负责让用户确认 prompt / negative_prompt
普通 Python：负责合并前端参数、签名、请求 LitMedia
```

这样做的好处是：

- LLM 不直接控制外部副作用。
- 只有 `is_ready=True` 后才提交。
- 接口调用逻辑和对话逻辑分离。

如果未来你要做 Web 版，可以让前端点击“生成”按钮后再调用接口，而不是 CLI 自动调用。

## 14. 当前脚本的设计优点

### 职责清晰

Agent 只负责内容创作，不负责 UI 参数。

### 可调试

结构化输出会打印出来，方便观察 AI 是否理解正确。

### 可扩展

未来可以增加：

- `ui_params_suggestion`
- prompt 质量评分
- 多语言翻译
- 风格模板
- 生成后轮询任务状态

### 状态可控

LangGraph 让每轮流程固定：

```text
先路由，再执行对应节点
```

不会让一个节点同时承担所有逻辑。

## 15. 你继续学习时该看哪里

建议按顺序看 `image_agent.py`：

1. 看 Pydantic 模型：理解输入输出协议。
2. 看 `build_llm()`：理解模型怎么接 Ark。
3. 看 `intent_router()`：理解怎么判断路由。
4. 看 `creative_worker()`：理解怎么让 AI 生成结构化结果。
5. 看 `build_graph()`：理解 LangGraph 怎么把节点连起来。
6. 看 `run_cli()`：理解每轮用户输入如何触发图执行。
7. 看 `call_litmedia_text_video()`：理解确认后如何调用真实接口。

掌握这几个点，就能理解这个 Agent 的完整运行过程。

## 16. 状态流转专题：`state = app.invoke(state)` 到底做了什么

这一节专门整理你刚才问到的几个关键点：

```python
state = app.invoke(state)
```

以及：

```python
"route": decision.route
```

和：

```python
route_by_intent(state)
```

### 16.1 `state` 是什么

在当前脚本里，`state` 是 LangGraph 每轮执行时传来传去的状态字典。

它的结构由 `CreativeState` 定义：

```python
class CreativeState(TypedDict):
    user_input: str
    params: CreativeParams
    chat_history: list[ChatTurn]
    route: Literal["chat", "creative"]
    reply: str
    is_ready: bool
    awaiting_confirmation: bool
```

初始化时在 `run_cli()` 里：

```python
state: CreativeState = {
    "user_input": "",
    "params": CreativeParams(),
    "chat_history": [],
    "route": "chat",
    "reply": "",
    "is_ready": False,
    "awaiting_confirmation": False,
}
```

可以把它理解成 Agent 当前这一轮的“工作台”：

```text
用户最新说了什么？
当前 prompt 是什么？
历史对话有哪些？
本轮要走 chat 还是 creative？
最终要回复用户什么？
是否已经确认生成？
```

### 16.2 每轮用户输入后，先更新 `state`

用户输入后，CLI 会先把最新输入写进状态：

```python
state = {
    **state,
    "user_input": user_input,
    "reply": "",
    "is_ready": False,
}
```

这里的 `**state` 表示复制旧状态。

然后后面的字段会覆盖旧值：

```python
"user_input": user_input
```

表示把用户这次新输入写入状态。

### 16.3 `app.invoke(state)` 是什么

`app` 是由：

```python
app = build_graph()
```

创建出来的 LangGraph 可执行对象。

所以：

```python
state = app.invoke(state)
```

意思是：

```text
把当前 state 交给 LangGraph 跑一遍状态机，
然后把 LangGraph 返回的新 state 覆盖回原来的 state。
```

你的图结构是：

```text
START
  ↓
intent_router
  ↓
根据 route 分支
  ├─ chat_responder
  └─ creative_worker
  ↓
END
```

因此 `app.invoke(state)` 实际会做：

1. 执行 `intent_router(state)`。
2. 拿到 `intent_router` 返回的新状态。
3. 调用 `route_by_intent(state)` 判断下一步。
4. 如果返回 `chat`，执行 `chat_responder(state)`。
5. 如果返回 `creative`，执行 `creative_worker(state)`。
6. 到 `END`。
7. 返回最终状态。

### 16.4 `route` 是在哪里写入 state 的

写入位置在 `intent_router()` 的返回值里：

```python
return {
    **state,
    "route": decision.route,
    "reply": "",
}
```

其中：

```python
decision = router_llm.invoke(messages)
```

是 AI 返回的路由判断。

例如 AI 返回：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

那么：

```python
"route": decision.route
```

就等价于：

```python
"route": "creative"
```

所以这个 return 实际是在说：

```python
return {
    **state,
    "route": "creative",
    "reply": "",
}
```

这就是把 AI 判断出来的路由写入 `state["route"]`。

### 16.5 `route_by_intent(state)` 是怎么用的

函数本身很简单：

```python
def route_by_intent(state: CreativeState) -> Literal["chat", "creative"]:
    return state["route"]
```

它只做一件事：

```text
读取 state["route"] 并返回。
```

真正使用它的是这段：

```python
graph.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "chat": "chat_responder",
        "creative": "creative_worker",
    },
)
```

含义是：

```text
intent_router 执行完以后，
LangGraph 调用 route_by_intent(state)，
如果返回 chat，就去 chat_responder，
如果返回 creative，就去 creative_worker。
```

### 16.6 一个完整例子

假设用户输入：

```text
我想做一个小狗视频
```

执行前状态可能是：

```python
state = {
    "user_input": "我想做一个小狗视频",
    "params": CreativeParams(prompt=None, negative_prompt=None),
    "chat_history": [],
    "route": "chat",
    "reply": "",
    "is_ready": False,
    "awaiting_confirmation": False,
}
```

然后执行：

```python
state = app.invoke(state)
```

第一步进入 `intent_router()`。

AI 返回：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

`intent_router()` 返回的新状态里：

```python
"route": "creative"
```

然后 LangGraph 调用：

```python
route_by_intent(state)
```

返回：

```python
"creative"
```

于是 LangGraph 进入：

```python
creative_worker(state)
```

`creative_worker` 可能返回：

```python
state = {
    "user_input": "我想做一个小狗视频",
    "params": CreativeParams(prompt=None, negative_prompt=None),
    "chat_history": [],
    "route": "creative",
    "reply": "我给你准备了 3 个小狗视频方向...",
    "is_ready": False,
    "awaiting_confirmation": False,
}
```

最后 `app.invoke(state)` 把这个最终状态返回给 CLI。

CLI 再打印：

```python
print(f"Agent: {state['reply']}")
```

### 16.7 为什么要写成 `state = app.invoke(state)`

因为 LangGraph 不会自动修改你外部变量里的旧 `state`。

它会返回一个“新状态”。

所以你必须写：

```python
state = app.invoke(state)
```

而不是只写：

```python
app.invoke(state)
```

如果只写 `app.invoke(state)`，图虽然执行了，但你没有保存返回的新状态，后续循环里就拿不到新的 `reply`、`params`、`awaiting_confirmation`。

### 16.8 这套模式的核心记法

可以记成：

```text
节点函数负责改 state
边负责决定下一个节点
app.invoke(state) 负责运行整张图
state = app.invoke(state) 负责保存运行后的结果
```

更短一点：

```text
state 进去，state 出来。
```

## 17. 每轮对话流程节点总览

这一节用更直观的方式，把“一轮用户输入”从终端进入，到 AI 处理，再到回复用户的完整流程写清楚。

## 17.1 总流程图

当前脚本每一轮对话的流程是：

```text
用户输入
  ↓
run_cli() 读取 input()
  ↓
把 user_input 写入 state
  ↓
state = app.invoke(state)
  ↓
LangGraph START
  ↓
intent_router
  ↓
route_by_intent(state)
  ↓
根据 state["route"] 分支
  ├─ chat      -> chat_responder  -> END
  └─ creative  -> creative_worker -> END
  ↓
app.invoke(state) 返回新 state
  ↓
run_cli() 打印 state["reply"]
  ↓
把 user / assistant 写入 chat_history
  ↓
如果 state["is_ready"] = True
  ↓
退出循环，调用 LitMedia 接口
```

最核心的一句话是：

```text
每轮对话 = 先路由，再执行对应业务节点，最后返回更新后的 state。
```

## 17.2 每个节点做什么

| 阶段 | 代码位置 | 是否调用 AI | 主要作用 | 主要修改的 state 字段 |
|---|---|---:|---|---|
| CLI 读取输入 | `run_cli()` | 否 | 读取用户输入，写入 `state["user_input"]` | `user_input`, `reply`, `is_ready` |
| 执行图 | `state = app.invoke(state)` | 间接调用 | 触发 LangGraph 按图执行 | 由节点决定 |
| 路由节点 | `intent_router()` | 是 | 判断当前输入是聊天还是创作 | `route`, `reply` |
| 分支函数 | `route_by_intent()` | 否 | 读取 `state["route"]`，决定下一节点 | 不修改 |
| 聊天节点 | `chat_responder()` | 是 | 处理能力介绍、闲聊、上下文问答 | `reply`, `is_ready` |
| 创作节点 | `creative_worker()` | 是 | 生成/修改 prompt，判断是否可提交 | `params`, `reply`, `is_ready`, `awaiting_confirmation` |
| 结束节点 | `END` | 否 | 本轮图执行结束 | 不修改 |
| 保存历史 | `run_cli()` | 否 | 把本轮 user/assistant 写入历史 | `chat_history` |
| 调接口 | `call_litmedia_text_video()` | 否 | 用户确认后请求生成接口 | 不属于 LangGraph |

## 17.3 第 1 步：CLI 读取用户输入

代码在 `run_cli()` 里：

```python
user_input = input("User: ").strip()
```

如果用户输入：

```text
我想生成一个关于小狗的视频
```

那么：

```python
user_input = "我想生成一个关于小狗的视频"
```

然后代码把它写入 `state`：

```python
state = {
    **state,
    "user_input": user_input,
    "reply": "",
    "is_ready": False,
}
```

这一步还没有调用 AI，只是更新状态。

## 17.4 第 2 步：调用 LangGraph

代码：

```python
state = app.invoke(state)
```

意思是：

```text
把当前 state 交给 LangGraph，让它按 build_graph() 定义的图跑一遍。
```

此时 LangGraph 会从 `START` 开始。

## 17.5 第 3 步：进入 `intent_router`

图里定义了：

```python
graph.add_edge(START, "intent_router")
```

所以第一站一定是：

```python
intent_router(state)
```

这个节点会组装给 AI 的上下文：

```python
messages = [
    SystemMessage(content=ROUTER_PROMPT),
    HumanMessage(
        content=(
            "当前内容参数：\n"
            f"{state['params'].model_dump_json(...)}\n\n"
            f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
            "最近对话：\n"
            f"{history_to_text(state['chat_history'], limit=8) or '暂无'}\n\n"
            "用户最新输入：\n"
            f"{state['user_input']}"
        )
    ),
]
```

然后调用路由 AI：

```python
decision = router_llm.invoke(messages)
```

AI 返回类似：

```json
{
  "route": "creative",
  "reason": "用户表达了视频创作需求"
}
```

然后写回 state：

```python
return {
    **state,
    "route": decision.route,
    "reply": "",
}
```

这里最重要的是：

```python
"route": decision.route
```

它把 AI 返回的 `"creative"` 写入了：

```python
state["route"]
```

## 17.6 第 4 步：条件分支

图里定义了：

```python
graph.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "chat": "chat_responder",
        "creative": "creative_worker",
    },
)
```

这表示：

```text
intent_router 执行完后，调用 route_by_intent(state)。
```

`route_by_intent` 是：

```python
def route_by_intent(state: CreativeState) -> Literal["chat", "creative"]:
    return state["route"]
```

它不调用 AI，也不改状态，只读：

```python
state["route"]
```

如果它返回：

```python
"chat"
```

LangGraph 走：

```text
chat_responder
```

如果它返回：

```python
"creative"
```

LangGraph 走：

```text
creative_worker
```

## 17.7 第 5A 步：如果走 `chat_responder`

适合这些输入：

```text
你能帮我干嘛？
你记得我刚刚说了什么吗？
我现在的 prompt 是什么？
```

代码会构造聊天上下文：

```python
messages = [
    SystemMessage(
        content=(
            f"{CHAT_PROMPT}\n\n"
            "当前内容参数：\n"
            f"{state['params'].model_dump_json(...)}"
        )
    )
]
```

然后把最近历史加入：

```python
for turn in state["chat_history"][-10:]:
    if turn.role == "user":
        messages.append(HumanMessage(content=turn.content))
    else:
        messages.append(AIMessage(content=turn.content))
```

最后追加本轮输入：

```python
messages.append(HumanMessage(content=state["user_input"]))
```

调用 AI：

```python
response = llm.invoke(messages)
```

写回 state：

```python
return {
    **state,
    "reply": str(response.content),
    "is_ready": False,
}
```

这个分支执行完后：

```python
graph.add_edge("chat_responder", END)
```

本轮结束。

## 17.8 第 5B 步：如果走 `creative_worker`

适合这些输入：

```text
我想生成一个关于小狗的视频
第二种
改成更搞笑一点
确认生成
```

代码会构造创作上下文：

```python
messages = [
    SystemMessage(content=CREATIVE_PROMPT),
    HumanMessage(
        content=(
            "当前内容参数：\n"
            f"{current_params.model_dump_json(...)}\n\n"
            "最近对话：\n"
            f"{history_to_text(state['chat_history'], limit=10) or '暂无'}\n\n"
            f"是否正在等待最终确认：{state['awaiting_confirmation']}\n\n"
            "用户最新输入：\n"
            f"{state['user_input']}"
        )
    ),
]
```

调用结构化 AI：

```python
understanding = creative_llm.invoke(messages)
```

这里的 `creative_llm` 是：

```python
creative_llm = llm.with_structured_output(CreativeUnderstanding)
```

所以 AI 必须返回 `CreativeUnderstanding` 结构。

例如：

```json
{
  "intent": "create_prompt",
  "params_patch": {
    "prompt": "短腿柯基在阳光客厅里追彩色毛绒球...",
    "negative_prompt": "画面昏暗、镜头剧烈晃动、场景杂乱"
  },
  "merge_strategy": "fill_missing",
  "reply": "我已经帮你整理成完整视频提示词啦。",
  "ready_for_submit": true,
  "confirm_submit": false
}
```

然后代码合并参数：

```python
updated_params = current_params.model_copy(deep=True)
patch = understanding.params_patch
allow_overwrite = understanding.merge_strategy == "overwrite"

if patch.prompt and (updated_params.prompt is None or allow_overwrite):
    updated_params.prompt = patch.prompt.strip()
```

如果用户已经确认提交：

```python
is_ready = bool(
    state["awaiting_confirmation"]
    and understanding.confirm_submit
    and updated_params.prompt
)
```

如果内容已经准备好，但还没最终确认：

```python
awaiting_confirmation = bool(
    understanding.ready_for_submit
    and updated_params.prompt
    and not is_ready
)
```

最后写回 state：

```python
return {
    **state,
    "params": updated_params,
    "reply": understanding.reply,
    "is_ready": is_ready,
    "awaiting_confirmation": awaiting_confirmation,
}
```

这个分支执行完后：

```python
graph.add_edge("creative_worker", END)
```

本轮结束。

## 17.9 第 6 步：`app.invoke(state)` 返回最终状态

不管走 `chat_responder` 还是 `creative_worker`，最后都会到：

```text
END
```

然后：

```python
state = app.invoke(state)
```

右边的 `app.invoke(state)` 返回最终状态。

左边的 `state =` 把新状态保存起来。

如果不写左边的 `state =`，就拿不到节点更新后的结果。

## 17.10 第 7 步：CLI 打印回复

图执行完后，CLI 打印：

```python
print(f"Agent: {state['reply']}")
```

这个 `reply` 来自：

- `chat_responder()` 的 `response.content`
- 或 `creative_worker()` 的 `understanding.reply`

## 17.11 第 8 步：保存对话历史

打印后，CLI 会保存本轮用户输入和助手回复：

```python
state["chat_history"].extend(
    [
        ChatTurn(role="user", content=user_input),
        ChatTurn(role="assistant", content=state["reply"]),
    ]
)
```

这一步非常重要。

下一轮调用 AI 时，`intent_router`、`chat_responder`、`creative_worker` 都会从 `chat_history` 里取最近对话作为上下文。

如果不保存历史，AI 就不知道上一轮发生了什么。

## 17.12 第 9 步：如果 `is_ready=True`，调用接口

循环条件是：

```python
while not state["is_ready"]:
```

如果 `creative_worker()` 返回：

```python
"is_ready": True
```

循环结束。

然后执行：

```python
result = call_litmedia_text_video(state["params"], frontend_params)
```

这里不再是 LangGraph 节点，而是普通 Python 业务逻辑。

参数来源：

```text
state["params"]：Agent 生成的 prompt / negative_prompt
frontend_params：前端选择的模型、比例、时长、清晰度、数量
```

合并后请求 LitMedia。

## 17.13 每轮节点输入输出速查表

| 节点 | 输入 | 输出 | 是否调用 AI | 典型用途 |
|---|---|---|---:|---|
| `intent_router` | 当前 `state` | 写入 `state["route"]` | 是 | 判断 chat/creative |
| `route_by_intent` | 当前 `state` | 返回 `"chat"` 或 `"creative"` | 否 | LangGraph 条件分支 |
| `chat_responder` | 当前 `state` | 写入 `state["reply"]` | 是 | 闲聊、能力介绍、上下文问答 |
| `creative_worker` | 当前 `state` | 写入 `params/reply/is_ready/awaiting_confirmation` | 是 | 创意构思、prompt 生成/修改、确认提交 |
| `call_litmedia_text_video` | `CreativeParams + FrontendParams` | 接口响应 | 否 | 最终生成视频 |

## 17.14 你可以这样记

```text
run_cli 负责循环
app.invoke 负责跑图
intent_router 负责分流
chat_responder 负责聊天
creative_worker 负责创作
call_litmedia_text_video 负责提交接口
```

更短的记法：

```text
输入 -> 写 state -> 跑图 -> 更新 state -> 打印 reply -> 保存历史 -> 必要时调接口
```
