# LangSmith 测试经验

这份文档记录当前视频创作 Agent 接入 LangSmith 后，怎么观察、怎么排查、怎么判断一次对话是不是正常。内容以 `image_agent.py` 的实际结构为准。

## 1. 先看项目是否接对

当前脚本里硬编码了：

```python
LANGSMITH_PROJECT_VALUE = "test"
```

所以 trace 应该出现在 LangSmith 的：

```text
Tracing -> test
```

如果看不到数据，先检查这几件事：

- 运行的是当前这份 `image_agent.py`。
- `LANGSMITH_API_KEY_VALUE` 已经填了真实 key。
- `configure_langsmith()` 在模型初始化之前执行。
- 页面左侧选的是 `Tracing`，不是 `Monitoring`。
- 项目名是 `test`，不是之前的 `litmedia-video-agent`。

LangSmith 有时会有几秒延迟，运行完一轮对话后可以刷新一下页面。

## 2. 一条 trace 对应什么

现在代码里每次执行：

```python
state = app.invoke(
    state,
    config={
        "run_name": "video_creation_agent",
        "tags": ["cli", "langgraph", "litmedia"],
        "metadata": {
            "route": state.get("route"),
            "awaiting_confirmation": state.get("awaiting_confirmation"),
            "has_prompt": bool(state["params"].prompt),
        },
    },
)
```

所以 LangSmith 里一条 `video_creation_agent` 基本对应一次用户输入。

比如用户连续输入三次：

```text
我想做一个小狗视频
第一个不错
确认生成
```

通常就会看到三条 `video_creation_agent` run。

## 3. 左侧树结构怎么看

点开一条 trace，左侧会出现类似结构：

```text
video_creation_agent
  ├─ intent_router
  │   └─ ChatOpenAI
  ├─ route_by_intent
  ├─ creative_worker
  │   └─ RunnableSequence
  │       └─ ChatOpenAI
  ├─ reflection_worker
  │   └─ ChatOpenAI
  ├─ route_after_reflection
  └─ tool_executor
```

不一定每次都有所有节点。

常见情况：

- 闲聊问题：通常是 `intent_router -> chat_responder`。
- 创作问题：通常是 `intent_router -> creative_worker -> reflection_worker`。
- 确认生成：如果工具调用没被反思节点拦截，会出现 `tool_executor`。

## 4. 每个节点重点看什么

### 4.1 `video_creation_agent`

这是整轮对话的总入口。

先看这里可以快速知道：

- 用户这一轮输入是什么。
- 最终输出给用户的回复是什么。
- 这轮有没有报错。
- 总耗时是多少。

如果只想快速判断这轮是否正常，先看最外层就够。

### 4.2 `intent_router`

这个节点负责判断路由：

```text
chat 或 creative
```

重点看 Output 里的：

```json
{
  "route": "creative",
  "reason": "..."
}
```

排查方式：

- 用户问“你能帮我干嘛”，应该是 `chat`。
- 用户说“我想做一个小狗视频”，应该是 `creative`。
- 用户说“确认生成”，如果正在等待确认，应该是 `creative`。

如果路由错了，优先改 `ROUTER_PROMPT`，不要急着改后面的节点。

### 4.3 `route_by_intent`

这个节点不调用模型，只读：

```python
state["route"]
```

它耗时通常接近 0。

如果它出问题，一般不是 AI 问题，而是 `intent_router` 写入的 `route` 不对，或者映射表没有覆盖对应 route。

### 4.4 `chat_responder`

这个节点处理闲聊、能力介绍、上下文问答。

重点看：

- Input 里的 `chat_history` 是否带上了最近对话。
- Input 里的 `params` 是否包含当前 prompt。
- Output 里的 `reply` 是否回答了用户的问题。

典型测试：

```text
你能帮我干嘛？
我刚刚说了什么？
现在的 prompt 是什么？
```

如果它答非所问，通常是上下文给得不够，或者 `CHAT_PROMPT` 约束不清。

### 4.5 `creative_worker`

这是最核心的创作节点。

重点看 Output：

```json
{
  "params": {
    "prompt": "...",
    "negative_prompt": "..."
  },
  "reply": "...",
  "awaiting_confirmation": true,
  "tool_call": {
    "tool_name": "none"
  }
}
```

检查点：

- 用户只说“小狗视频”时，不应该直接提交，应该给创意方向。
- 用户选择“第一个”时，应该整理出完整 prompt。
- 用户说“改成更搞笑”，应该覆盖旧 prompt。
- 用户确认生成时，才应该出现 `tool_call.tool_name = submit_video_generation`。

如果 prompt 太泛，就改 `CREATIVE_PROMPT` 的创作标准。

如果工具调用太早，就改 `CREATIVE_PROMPT` 里关于 `tool_call` 的规则。

### 4.6 `reflection_worker`

这个节点是安全检查层。

它负责判断：

- prompt 是否太泛。
- 用户是否真的确认提交。
- reply 有没有提前说“已经生成成功”。
- tool_call 是否应该被阻止。

重点看 Output：

```json
{
  "reflection_passed": true,
  "reflection_issues": [],
  "tool_call": {
    "tool_name": "none"
  }
}
```

如果它阻止了工具调用，通常会看到：

```json
{
  "should_block_tool": true,
  "issues": ["用户尚未明确确认生成"]
}
```

这个节点很适合排查“为什么没有调用接口”。

### 4.7 `tool_executor`

只有 Agent 决定调用工具，且反思节点没有阻止时，才会进入这里。

重点看：

- `tool_name` 是否是 `submit_video_generation`。
- tool 入参里的 `prompt` 是否正确。
- 前端 UI 参数是否合并正确。
- 接口返回是否成功。

如果没看到 `tool_executor`，说明还没真正提交。

排查顺序：

```text
creative_worker 是否返回 tool_call？
reflection_worker 是否拦截？
route_after_reflection 是否走 tool？
```

## 5. Input / Output 怎么看

每个节点右侧都有 `Input` 和 `Output`。

### Input

Input 是节点执行前拿到的状态。

常见字段：

```text
user_input
params
frontend_params
chat_history
awaiting_confirmation
tool_call
```

看 Input 是为了确认：

- 上一轮状态有没有保存下来。
- 当前 prompt 是否已经存在。
- 当前是不是等待确认状态。
- 前端参数是否正确。

### Output

Output 是节点执行后返回的新状态。

看 Output 是为了确认：

- 这一步有没有更新 prompt。
- 有没有生成 reply。
- 有没有设置 awaiting_confirmation。
- 有没有设置 tool_call。
- 有没有进入 is_ready。

## 6. 什么时候看 ChatOpenAI 子节点

如果只看业务流程，通常看父节点 Output 就够。

需要深入排查模型输入输出时，再点 `ChatOpenAI`：

- Input：真实发给模型的 messages。
- Output：模型原始响应。

尤其是排查这些问题时要看 `ChatOpenAI`：

- AI 为什么判断成 `chat`。
- AI 为什么没有提取 prompt。
- AI 为什么没有设置 `tool_call`。
- AI 为什么误以为已经确认。

注意：`creative_worker` 使用了 `with_structured_output`，所以 LangSmith 里可能会显示：

```text
RunnableSequence
  └─ ChatOpenAI
```

这是正常的。结构化输出内部会有一层 runnable 包装。

## 7. 常见问题排查

### 7.1 看不到 trace

检查：

- 项目名是不是 `test`。
- `LANGSMITH_API_KEY_VALUE` 有没有填真实 key。
- 是否重新启动了脚本。
- 是否在 `Tracing` 页面，而不是 `Monitoring` 页面。

### 7.2 route 错了

看：

```text
intent_router -> Output
```

如果 `route` 错，改：

```python
ROUTER_PROMPT
```

不要先改 `creative_worker`。

### 7.3 prompt 没更新

看：

```text
creative_worker -> Output -> params
```

如果 `params.prompt` 没变，再看：

```text
creative_worker -> ChatOpenAI -> Output
```

判断是模型没返回，还是合并逻辑没覆盖。

### 7.4 工具没调用

按顺序看：

```text
creative_worker -> tool_call
reflection_worker -> should_block_tool
route_after_reflection
tool_executor 是否出现
```

常见原因：

- 用户没有明确确认。
- prompt 太泛，被反思节点拦截。
- `tool_call.tool_name` 是 `none`。

### 7.5 工具调用了但接口失败

看：

```text
tool_executor -> Output
```

以及：

```text
LitMedia payload
```

重点检查：

- token 是否有效。
- signature 是否生成。
- prompt 是否为空。
- UI 参数是否合理。
- 接口返回的错误 body。

## 8. 建议的测试用例

每次改 prompt 或节点逻辑后，建议固定跑这些用例。

### 闲聊

```text
你能帮我干嘛？
```

预期：

```text
route = chat
不进入 creative_worker
不调用 tool
```

### 泛创意

```text
我想做一个小狗视频
```

预期：

```text
route = creative
给 2-3 个方向
不设置 tool_call
```

### 选择方案

```text
第一个不错
```

预期：

```text
生成完整 prompt
awaiting_confirmation = true
不调用 tool
```

### 修改 prompt

```text
改成更搞笑一点
```

预期：

```text
prompt 被改写
reply 明确说明已修改
不调用 tool
```

### 确认生成

```text
确认生成
```

预期：

```text
creative_worker 返回 tool_call = submit_video_generation
reflection_worker 通过
进入 tool_executor
```

### 未确认时问状态

```text
提交了吗？
```

预期：

```text
不调用 tool
回复当前还没有提交
```

## 9. 推荐看 trace 的顺序

一条 trace 最有效的阅读顺序：

```text
1. video_creation_agent 总输入输出
2. intent_router 看 route 是否正确
3. creative_worker 看 prompt/reply/tool_call
4. reflection_worker 看是否拦截
5. tool_executor 看接口入参和响应
```

不要一上来就看最底层 ChatOpenAI。先看业务节点 Output，通常更快。

## 10. 一句话经验

LangSmith 不是只看“模型说了什么”，而是看：

```text
状态是怎么变的，
节点是怎么走的，
工具为什么调用或没调用。
```

对你这个 Agent 来说，最重要的三个字段是：

```text
route
params.prompt
tool_call.tool_name
```

只要这三个字段符合预期，大部分流程就是正常的。
