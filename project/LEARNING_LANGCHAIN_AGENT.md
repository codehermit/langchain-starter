## LangChain 1.0 Agent 学习路线与实战计划

> 本文档放在项目中，作为你系统学习 LangChain Agent 开发的“学习手册”。  
> 当前仓库示例使用 Python + LangChain 1.0 + 环境变量配置（`.env`）。

---

## 0. 起点：从 `01/hello.py` 开始

- **文件**：`project/01/hello.py`
- **作用**：完成“模型能跑起来”的最小示例。
- **要点**：
  - 通过 `load_dotenv()` 从 `.env` 中读取 `API_KEY`、`BASE_URL`。
  - 使用 `init_chat_model` 初始化模型并调用 `model.invoke()`。

**建议改进**（可选）：

- 将模型初始化封装成函数，例如：
  - `get_model()`：返回全局复用的模型实例。
- 将一次对话封装成函数，例如：
  - `simple_chat(prompt: str) -> str`

这样后续其它示例可以直接复用，减少重复代码。

---

## 1. 阶段一：掌握 LangChain 1.0 基础（1–2 天）

### 1.1 学习目标

- 熟悉 LangChain 1.0 的几个核心概念：
  - **LCEL（LangChain Expression Language）**
  - **Runnable 接口：`invoke / batch / stream`**
  - **Prompt 模板：`ChatPromptTemplate`**
- 能基于这些概念写出 **可维护的聊天脚本**。

### 1.2 建议新文件

1. **`project/02/basic_chat.py`**：模板化单轮对话

   - 使用 `ChatPromptTemplate` 组合：
     - 系统角色：如“你是一个耐心的中文讲解助手”。
     - 用户输入：从命令行 `input()` 获取。
   - 用 `model.invoke()` 得到回复并打印。

2. **`project/02/multi_turn_manual.py`**：手工多轮对话（无 Memory）
   - 维护一个简单的 `history` 列表：
     - 存储若干轮对话 `[(role, content), ...]`。
   - 每次调用时手工把历史拼接/传入 Prompt。
   - 目的是体会：“**上下文都是通过 prompt 带给模型的**”。

### 1.3 本阶段 checklist

- [ ] 理解并能使用 `ChatPromptTemplate`。
- [ ] 能写出支持命令行输入的问答脚本。
- [ ] 能手工拼接多轮对话的上下文。

---

## 2. 阶段二：RAG 与本地知识库问答（3–5 天）

智能客服几乎都需要“看文档再回答”，这就是 **RAG（Retrieval-Augmented Generation）**。

### 2.1 要掌握的概念

- **文档加载（Document Loaders）**
  - 从本地 `txt / md / pdf / docx` 等加载文本成 `Document` 对象。
- **文本分块（Text Splitter）**
  - 将长文档切分为合适大小的 chunk，便于向量化和检索。0
- **向量存储与检索器（Vector Store & Retriever）**
  - 常用：`FAISS`、`Chroma` 等。
  - 将文档向量化后存入向量库，通过相似度检索相关片段。
- **RAG 链**
  - 用户问题 → 检索相关文档 → 将“文档片段 + 问题”交给模型 → 生成答案。

### 2.2 建议目录与文件

- 新建目录：`docs/`，例如：

  - `docs/faq.md`：模拟公司 FAQ 文档（退款、发货、售后等常见问题）。

- 新建脚本：`project/03/rag_qa.py`
  - **功能**：
    - 第一次运行：从 `docs/faq.md` 加载文档 → 分块 → 建立向量库（如 FAISS），保存到本地（如 `data/faiss_index`）。
    - 后续运行：如果本地已有向量库，则直接加载，节省时间。
    - 命令行输入问题：
      - 使用 retriever 检索最相关的几个 chunk。
      - 将检索结果和问题一起丢给模型，让模型根据文档回答。
      - 可选：打印出模型使用的文档片段，方便验证。

### 2.3 本阶段 checklist

- [ ] 能从本地文档加载文本并分块。
- [ ] 能创建一个向量库，并通过问题检索相关文档。
- [ ] 能实现一个最小的 RAG 问答脚本。

---

## 3. 阶段三：Tools & Agent 基础（3–5 天）

Agent 的本质：**大模型 + 一组工具（Tools） + 决策逻辑**。  
模型不再只是“回答”，而是可以决定 **“要不要调用工具、调用哪个工具、如何使用工具结果”**。

### 3.1 要掌握的概念

- **Tool（工具）**
  - 本质是一个带有签名和说明的函数，供模型调用。
  - 示例工具：
    - 时间工具：返回当前时间。
    - 计算器：执行加减乘除。
    - HTTP 请求工具：访问第三方接口。
- **Agent & AgentExecutor**
  - 在 LangChain 1.0 中，用官方推荐的 agent 创建方法（名称可能是 `create_react_agent`、`create_tool_calling_agent` 等，以文档为准）。
  - 流程：
    1. 用户输入任务。
    2. 模型分析：是否需要工具？需要哪个工具？
    3. 调用工具 → 获得结果。
    4. 再次思考，并给出最终回答（或者继续调用工具）。

### 3.2 建议新文件：`project/04/tools_agent.py`

- 定义至少 3 个工具：
  1. `get_current_time()`：返回当前时间字符串。
  2. `simple_calculator(expression: str)`：处理基础算术表达式。
  3. `mock_weather(city: str)`：返回某城市的“模拟天气”（可以直接写固定文案）。
- 定义一个 Agent：
  - 将上述工具注册给 Agent。
  - 让 Agent 根据用户需求自动决定是否调用工具。
- 支持命令行交互：
  - 用户可以输入：
    - “现在几点了？”
    - “帮我算一下 1314 × 520”
    - “今天杭州天气如何？”
  - 观察 Agent 的思考过程（可选：开启 verbose，打印中间步骤）。

### 3.3 本阶段 checklist

- [ ] 理解 Tool 的概念并能封装 Python 函数为 Tool。
- [ ] 能创建一个能自动选择工具的 Agent。
- [ ] 能在命令行和 Agent 进行多轮交互。

---

## 4. 阶段四：RAG + Agent 打造“客服 Agent 原型”（5–7 天）

这一步是从“技术 Demo”变成“有业务价值的客服雏形”。

### 4.1 设计思路

- 把前面做好的 **RAG 问答** 封装成一个 Tool：
  - 例如：`faq_rag_tool(question: str) -> str`。
  - 内部流程：检索 FAQ 文档 → 将相关内容 + 问题交给模型 → 返回答案。
- 再增加一些“业务工具”（先用 mock 数据即可）：
  - `query_order_status(order_id: str) -> str`：返回虚拟的订单状态。
  - `query_shipping_info(order_id: str) -> str`：返回虚拟的物流信息。

然后：

- 使用 Agent 把这些工具组合起来，让 Agent 来决定使用哪种工具/组合。

### 4.2 建议新文件：`project/05/cust_service_agent_cli.py`

- **目标**：命令行版智能客服。
- **功能示例**：
  - 用户输入：
    - “如何申请退款？” → Agent 调用 FAQ RAG Tool。
    - “帮我查一下订单 123456 的物流情况” → Agent 调用订单/物流相关工具。
  - Agent 返回自然语言答案，并且在回答中“解释自己做了什么”（可通过系统提示词控制风格）。

### 4.3 本阶段 checklist

- [ ] 已有 FAQ 文档 + RAG 工具。
- [ ] 已有模拟的订单/物流工具。
- [ ] Agent 能够在命令行中扮演“智能客服”的角色。

---

## 5. 阶段五：对话记忆（更像真人客服）（2–3 天）

客服场景中，多轮对话的上下文非常关键。

### 5.1 要掌握的概念

- **Memory（记忆）**
  - 短期记忆：一段会话中的最近几轮对话。
  - 可以只保留 N 轮，以控制 Token 成本。
- **Agent + Memory**
  - 让 Agent 在多轮对话中记得：
    - 用户的昵称/偏好。
    - 当前正在处理的订单号等。

### 5.2 建议改造：升级 `project/05/cust_service_agent_cli.py`

- 为 Agent 加上会话记忆：
  - 用户先说：“帮我查一下订单 123456 的物流情况。”
  - 再问：“那大概什么时候能到？”（不重复订单号）
  - Agent 仍然知道是在问 123456 的物流。

### 5.3 本阶段 checklist

- [ ] 掌握 LangChain 的 Memory 用法。
- [ ] 让客服 Agent 在一轮会话内记住关键上下文信息。

---

## 6. 阶段六：评估与监控（上线前的“质量保障”）（3–5 天）

上线之前要能 **评估系统质量、监控问题**。

### 6.1 评估（Evaluation）

- 准备一批测试问题与期望答案/标签：
  - 如：`tests/cases.json`。
- 使用脚本批量调用你的客服 Agent：
  - 记录 Agent 的回答。
  - 人工抽查，或使用额外模型做自动打分（可选）。

### 6.2 建议新文件：`project/eval/eval_cust_service.py`

- **功能**：
  - 批量读取测试用例（问题、期望标签）。
  - 依次调用你的客服 Agent。
  - 输出：
    - 简单的统计信息（回答长度、是否提到某些关键词等）。
    - 以及原始问答日志，方便人工评估。

### 6.3 本阶段 checklist

- [ ] 能批量测试客服 Agent 的表现。
- [ ] 能通过日志与简单统计，发现系统的薄弱点。

---

## 7. 阶段七：Web API & 简单前端（真正“上线可用”）（5–7 天）

### 7.1 后端 API（推荐 FastAPI）

- 新建文件：`project/06/api_server.py`
- **目标**：把客服 Agent 封装成 HTTP 接口：
  - `POST /chat`：
    - 请求体：`{ "session_id": "xxx", "message": "用户问题" }`
    - 返回：`{ "reply": "机器人回答", "session_id": "xxx", "trace_id": "..." }`
  - 可选：`GET /health` 用于健康检查。
- 注意：
  - 按 `session_id` 管理会话记忆，每个会话独立。

### 7.2 前端界面（可选技术：Streamlit / React / Vue）

- 如果要快速上手：
  - 可以用 `Streamlit` 写 `project/07/web_ui.py`：
    - 左边选择会话，右边是聊天窗口。
    - 每条用户消息通过 HTTP 调用 `POST /chat`。
- 如果偏向工程化前端：
  - 新建 `project/07/web_ui/` 前端项目，用你熟悉的框架：
    - React + Vite / Next.js
    - 或 Vue + Vite，等。

### 7.3 本阶段 checklist

- [ ] 已有可调用的 HTTP 后端服务。
- [ ] 至少有一个简单 Web 聊天界面。
- [ ] 能在本机或局域网访问到这个客服系统。

---

## 8. 阶段八：部署与运维（2–5 天）

### 8.1 部署建议

- **轻量方案**：
  - 购买云服务器（阿里云 / 腾讯云 / AWS 等）。
  - 安装 Python 环境，在服务器上运行 `uvicorn` 启动 FastAPI。
  - 用 `nginx` 做反向代理与 HTTPS（可选）。
- **容器化方案**（推荐）：
  - 编写 `Dockerfile`。
  - 使用 `docker-compose` 管理服务（应用 + 向量库 + 反向代理）。

### 8.2 日志与监控

- 记录内容：
  - 时间戳、会话 ID、用户问题、系统回答。
  - 工具调用情况、错误堆栈、接口耗时。
- 通过日志你可以：
  - 发现哪些问题回答得不好 → 调整 RAG/Prompt。
  - 估计系统负载 → 确定扩容与缓存策略。

---

## 9. 建议的项目结构示例

随着你一步步实现上述阶段，项目目录可以逐渐演化为：

- `project/`
  - `01/hello.py`（最基础模型调用）
  - `02/basic_chat.py`（模板化单轮对话）
  - `02/multi_turn_manual.py`（手工多轮对话）
  - `03/rag_qa.py`（本地知识库问答）
  - `04/tools_agent.py`（工具型 Agent）
  - `05/cust_service_agent_cli.py`（命令行客服 Agent，带记忆）
  - `06/api_server.py`（对外提供 HTTP API）
  - `07/web_ui/`（前端或 Streamlit）
- `docs/`
  - `faq.md`（FAQ 知识库示例）
- `eval/`
  - `eval_cust_service.py`（评估脚本）
  - `tests/`（测试用例）
- 其他：
  - `.env`（API Key、BASE_URL 等配置）
  - `requirements.txt`（Python 依赖）
  - `README.md`（项目简介与运行说明）
  - `LEARNING_LANGCHAIN_AGENT.md`（本学习文档）

---

## 10. 使用建议

- 每完成一个阶段或一个示例文件：
  - 在本学习文档中勾选对应的 checklist。
  - 在代码中多写注释，记录自己的理解。
- 如果你对某个概念（如 RAG、Tool、Agent、Memory 等）有疑问：
  - 可以在此文档末尾开一个“疑问记录”小节，写下问题。
  - 然后再通过文档、官方教程、或与助手对话来逐条解决。

> 后续如需，我可以根据你已经完成的进度，进一步为其中某个阶段补充 **详细代码模板**（例如直接给出 `03/rag_qa.py` 的示例实现），你只需告诉我当前已实现到哪一步即可。
