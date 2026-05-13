# Video Agent Framework

基于 LangGraph + LangChain 的视频创作 Agent 框架。  
当前版本支持：

- 多轮会话状态管理（`CreativeState`）
- AI 路由（chat/creative）
- AI 判断是否需要 RAG（检索路由）
- 本地知识库检索（`knowledge/*.md`）
- AI 长期记忆写入（`memory/user_memory.json`）
- 反思审查与工具调用（LitMedia 文生视频）

---

## 1. 快速开始

### 1.1 环境要求

- Python 3.10+
- 已安装依赖：
  - `langgraph`
  - `langchain-core`
  - `langchain-openai`
  - `pydantic`
  - `python-dotenv`（可选）

### 1.2 安装依赖

```bash
pip install langgraph langchain-core langchain-openai pydantic python-dotenv
```

### 1.3 配置环境变量

将仓库根目录的 `.env.example` 复制为 `.env` 并填写真实值（`.env` 已被忽略，不会进入 Git）。

最少需要：

```env
ARK_API_KEY=你的ArkKey
ARK_MODEL=ark-code-latest
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
```

可选：

```env
DEBUG_AI_OUTPUT=1
RAG_TOP_K=3
RAG_CHUNK_MAX_CHARS=900
LITMEDIA_API_SECRET=...
LITMEDIA_DEVICE_CODE=...
LITMEDIA_TOKEN=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

### 1.4 启动方式

推荐：

```bash
python -m video_agent
```

兼容旧入口：

```bash
python image_agent.py
```

---

## 2. 项目结构

```text
lang/
  README.md
  image_agent.py                  # 兼容入口（转发到 video_agent.cli）
  knowledge/                      # 本地 RAG 知识库
    prompt_templates.md
    negative_prompt_examples.md
    video_style_guides.md
  memory/
    user_memory.json              # 长期记忆文件
  video_agent/
    __main__.py                   # python -m video_agent
    __init__.py                   # 导出 build_graph / run_cli
    cli.py                        # 终端交互入口
    settings.py                   # 全局配置与路径
    logging_utils.py              # debug_print
    prompts.py                    # 所有系统提示词
    models.py                     # Pydantic 协议 + CreativeState
    llm_factory.py                # ChatOpenAI 与结构化 LLM
    memory_store.py               # 长期记忆读写/迁移/格式化
    rag_service.py                # 轻量本地 RAG
    litmedia_client.py            # LitMedia HTTP 与 Tool
    graph/
      builder.py                  # 状态机装配
      nodes.py                    # 节点实现
      routing.py                  # 条件边路由函数
      utils.py                    # 通用工具函数
```

---

## 3. 架构设计

### 3.1 分层原则

- **Model 层**：协议与状态定义（`models.py`）
- **Prompt 层**：提示词配置集中管理（`prompts.py`）
- **Service 层**：外部能力（RAG、Memory、HTTP Tool）
- **Graph 层**：业务流程编排（节点 + 路由 + builder）
- **Entry 层**：CLI/未来 Web API

### 3.2 为什么要拆分

- 单文件难维护，改动容易互相污染
- 提示词、模型协议、业务流程应解耦
- 便于测试（可单测节点/服务，不必整图联调）
- 便于扩展（替换 RAG 后端、改 DB、加 API）

---

## 4. 核心状态机流程

每轮输入执行一次图：

```text
START
  -> memory_retriever
  -> retrieval_router
       ├─ retrieve -> rag_retriever
       └─ skip --------------------\
  -> intent_router                 |
       ├─ chat -> chat_responder --|
       └─ creative -> creative_worker -> reflection_worker
                                         ├─ tool -> tool_executor
                                         └─ end
  -> memory_writer
  -> END
```

### 4.1 关键分支

- **RAG 分支**：`retrieval_router` 先判断要不要检索
- **意图分支**：`intent_router` 判断聊天/创作
- **工具分支**：`reflection_worker` 决定是否调用工具

---

## 5. 核心模块说明

### `video_agent/models.py`

- 定义所有结构化协议（Pydantic）
- 定义全局状态 `CreativeState`
- 关键模型：
  - `CreativeUnderstanding`
  - `RetrievalDecision`
  - `MemoryWriteDecision`
  - `ToolCallRequest`

### `video_agent/llm_factory.py`

- 创建基础 `ChatOpenAI` 客户端
- 统一创建结构化输出 LLM：
  - `router_llm`
  - `creative_llm`
  - `reflection_llm`
  - `retrieval_llm`
  - `memory_llm`

> 当前统一使用 `method="function_calling"`，避免部分模型不支持 `json_schema` 的问题。

### `video_agent/graph/nodes.py`

- 图节点的核心业务逻辑都在这里
- 节点是纯状态变换函数：输入 `state`，输出新 `state`

### `video_agent/rag_service.py`

- 本地 Markdown 检索实现（轻量）
- 检索流程：切片 -> 分词 -> 匹配打分 -> TopK -> 拼装上下文

### `video_agent/memory_store.py`

- 统一记忆结构：`{"memories": [...]}`（条目式）
- 支持旧结构自动迁移（legacy arrays -> records）
- 基于 `category + value` 生成稳定 ID，支持幂等更新

### `video_agent/litmedia_client.py`

- LitMedia 签名生成与 HTTP 调用
- 暴露 `submit_video_generation` 作为 Tool 节点调用

### `video_agent/cli.py`

- 终端会话驱动器
- 按轮调用 `app.invoke(state)`，维护对话历史

---

## 6. 长期记忆机制

### 6.1 文件格式

`memory/user_memory.json`：

```json
{
  "memories": []
}
```

每条记录结构（示例）：

```json
{
  "id": "mem_xxx",
  "category": "video_style",
  "value": "电影感",
  "confidence": 0.95,
  "reason": "用户明确表达偏好",
  "source_text": "我喜欢电影感",
  "created_at": "2026-05-13T11:20:45+0800",
  "updated_at": "2026-05-13T11:20:45+0800"
}
```

### 6.2 写入策略

- 由 `memory_writer` 使用 AI 结构化判断
- 仅接受高置信度（当前阈值 `>= 0.75`）
- Python 端做去重、合并、限流（confirmed_prompt 最多 5 条）

---

## 7. RAG 机制

### 7.1 是否检索

- 由 `retrieval_router` 用 AI 判定 `should_retrieve`
- 若 `false` 直接跳过 RAG，减少延迟与噪音

### 7.2 检索数据源

- 本地目录：`knowledge/*.md`
- 轻量关键词匹配（可后续替换为向量库）

### 7.3 检索输入

- 优先使用 AI 生成的 `retrieval_query`
- 兜底使用用户原始输入

---

## 8. 可观测性与调试

### 8.1 本地调试

- `DEBUG_AI_OUTPUT=1` 时打印每个关键节点输出
- 建议先观察：
  - `retrieval_router AI 判断`
  - `intent_router AI 返回`
  - `creative_worker AI 返回`
  - `memory_writer AI 判断`

### 8.2 LangSmith

在环境中设置 `LANGSMITH_API_KEY`（可选 `LANGSMITH_PROJECT`、`LANGSMITH_ENDPOINT`）后，`settings.py` 会在启动时写入 LangSmith 相关环境变量以启用 tracing。  
未设置密钥时不会启用，避免把密钥写进代码仓库。  
运行后可在 LangSmith 中按节点查看状态流与模型输入输出。

---

## 9. 常见扩展方向

### 9.1 换成向量 RAG

- 保持 `rag_retriever` 节点接口不变
- 只替换 `rag_service.py` 实现

### 9.2 长期记忆落到数据库

- 保持 `memory_retriever/memory_writer` 节点接口不变
- 只替换 `memory_store.py` 持久化层

### 9.3 增加 Web API

- 复用 `graph/builder.py` 与 `graph/nodes.py`
- 新建 `api.py`，把 `run_cli` 循环改为 HTTP 请求驱动

### 9.4 多模型策略

- 在 `llm_factory.py` 拆分不同节点模型
- 例如路由模型用快模型，创作模型用强模型

---

## 10. 开发约定

- 新增业务能力优先落在 `graph/nodes.py`（节点）或 service 层（外部能力）
- 不建议把业务逻辑重新塞回 `image_agent.py`
- 新增提示词统一放 `prompts.py`
- 新增协议统一放 `models.py`

---

## 11. 迁移说明（旧版单文件）

- 原 `image_agent.py` 已简化为兼容入口
- 实现迁移到 `video_agent` 包
- 旧启动命令仍可用，不影响使用

---

## 12. License / Internal Notes

本仓库目前未附带开源 License。若需开源，请补充许可协议并清理敏感配置。
