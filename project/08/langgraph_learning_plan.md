# LangGraph 学习路线

本文档以 `basic_langgraph.py` 为起点，按“先看懂最小图，再扩展成真实 Agent 工作流”的顺序学习 LangGraph。

## 0. 当前代码对应的核心概念

`basic_langgraph.py` 已经覆盖了 LangGraph 入门中最关键的一条主线：

```text
用户输入
  -> call_model
  -> 判断是否需要工具
  -> tools / END
  -> call_model
  -> END
```

对应概念如下：

| 代码位置                       | LangGraph 概念      | 要理解的问题                               |
| ------------------------------ | ------------------- | ------------------------------------------ |
| `AgentState`                   | State               | 图运行时传递什么数据                       |
| `Annotated[..., add_messages]` | Reducer             | 多个节点如何合并状态更新                   |
| `call_model`                   | Node                | 节点本质上是接收 state、返回局部更新的函数 |
| `should_continue`              | Conditional routing | 如何根据当前状态决定下一步                 |
| `ToolNode(TOOLS)`              | Tool execution      | 模型提出工具调用，图负责执行               |
| `build_graph()`                | Graph construction  | 节点和边如何组成可执行图                   |
| `graph_app.invoke(state)`      | Runtime             | 图如何从入口运行到结束                     |

学习时先不要急着做复杂多 Agent。LangGraph 最重要的直觉是：它不是一个“自动帮你写 Agent 的黑盒”，而是一个“可持久化、可观测、可分支、可循环的状态机框架”。

## 1. 第一阶段：看懂最小图

目标：理解 LangGraph 的最小执行单元。

重点：

- `StateGraph(AgentState)`：声明一个图，图的状态类型是 `AgentState`
- `graph.add_node(...)`：注册节点
- `graph.set_entry_point(...)` 或 `START`：声明入口
- `graph.add_edge(...)`：声明固定流转
- `graph.compile()`：编译成可运行对象
- `invoke(...)`：执行一次完整图流程

练习：

1. 把当前图改成无工具版本：`call_model -> END`
2. 暂时不用真实模型，让 `call_model` 返回固定 `AIMessage`
3. 打印每次进入节点时的 `state`
4. 画出当前图的流程图

验收标准：

- 能解释“节点返回的是状态更新，不是完整状态”
- 能独立写出一个 `START -> node -> END` 的最小图

## 2. 第二阶段：吃透 State 和 Reducer

目标：理解状态如何在多个节点之间传递和合并。

重点：

- `state` 是整个图共享的数据上下文
- 每个节点接收完整 state，返回局部更新
- 没有 reducer 的字段通常会被新值覆盖
- 有 reducer 的字段可以自定义合并逻辑
- `add_messages` 会把新消息追加到历史消息中

建议新建一个练习文件，例如 `state_reducer_demo.py`：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class DemoState(TypedDict):
    count: int
    messages: Annotated[list[AnyMessage], add_messages]
```

练习：

1. 节点 A 把 `count` 加 1
2. 节点 B 往 `messages` 里追加一条消息
3. 多运行几轮，观察 `count` 和 `messages` 的变化
4. 去掉 `add_messages`，比较消息字段的变化

验收标准：

- 能说清楚 reducer 解决的是什么问题
- 能判断某个字段应该“覆盖”还是“累加”

## 3. 第三阶段：掌握条件路由

目标：理解 Agent 的“智能流转”本质上是条件边。

当前代码中的核心是：

```python
graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
```

重点：

- 条件函数只负责判断下一步，不负责执行业务逻辑
- 条件函数返回值要和映射表 key 对应
- 条件边可以实现分支、循环、提前终止

练习：

1. 用户输入包含“计算”时进入 `tools`
2. 用户输入包含“总结”时进入 `summarize`
3. 其他情况直接结束
4. 增加一个 `debug` 节点打印当前状态

验收标准：

- 能独立写出一个三分支图
- 能解释循环边为什么不会自动无限执行，而是由状态和条件控制

## 4. 第四阶段：工具调用机制

目标：理解模型、工具和 LangGraph 各自的职责。

当前代码中的完整链路是：

1. `@tool` 定义工具
2. `_model.bind_tools(TOOLS)` 告诉模型有哪些工具
3. 模型返回 `tool_calls`
4. `should_continue` 检查是否存在工具调用
5. `ToolNode(TOOLS)` 真正执行工具
6. 工具结果写回 `messages`
7. 回到 `call_model`，模型基于工具结果生成最终回答

重点：

- 模型不会直接执行工具
- 模型只是生成工具调用请求
- `ToolNode` 才是真正执行工具的节点
- 工具执行结果会作为消息回到上下文中

练习：

1. 增加 `get_current_time` 工具
2. 增加 `unit_convert` 工具
3. 增加 `search_local_note(keyword)` 假工具
4. 在 `tools` 节点前后打印工具调用名称和参数

验收标准：

- 能解释 `tool_calls` 和工具执行结果的关系
- 能独立增加一个工具并让模型调用

## 5. 第五阶段：多节点 Workflow

目标：从“一个模型节点 + 一个工具节点”扩展到更真实的业务流程。

建议做一个研究助手：

```text
question
  -> planner
  -> researcher
  -> writer
  -> reviewer
  -> END
```

可以把状态设计成：

```python
class ResearchState(TypedDict):
    question: str
    plan: str
    documents: list[str]
    draft_answer: str
    review_result: str
    need_revision: bool
```

重点：

- 不同节点可以使用不同 prompt
- 不同节点可以负责不同职责
- 状态不一定只有 `messages`
- 条件边可以用于“返工”和“重试”

练习：

1. `planner` 负责生成研究计划
2. `researcher` 负责收集材料
3. `writer` 负责生成答案
4. `reviewer` 判断是否需要修改
5. 如果 `need_revision=True`，回到 `writer`

验收标准：

- 能拆分一个复杂任务为多个节点
- 能设计每个节点的输入、输出和状态字段

## 6. 第六阶段：持久化、记忆和多轮会话

目标：理解 LangGraph 相比普通 chain 更适合长流程 Agent 的原因。

重点：

- checkpoint
- thread_id
- 状态恢复
- 多轮会话
- human-in-the-loop
- time travel

练习：

1. 给图增加 checkpointer
2. 使用不同 `thread_id` 保存不同会话
3. 中断一次执行，再从中断处恢复
4. 在关键节点暂停，等待人工确认

验收标准：

- 能解释 checkpoint 和普通聊天历史的区别
- 能用 `thread_id` 管理多个会话

## 7. 第七阶段：流式输出和调试

目标：让图执行过程可观察。

重点：

- `stream`
- `astream`
- 节点级事件
- state 更新事件
- token 级输出
- LangSmith tracing

练习：

1. 把 `invoke` 改成 `stream`
2. 每经过一个节点就打印节点名
3. 打印每个节点返回的局部状态更新
4. 对模型输出做流式展示

验收标准：

- 能知道一次图执行经过了哪些节点
- 能定位状态在哪个节点发生了变化

## 8. 第八阶段：实战项目路线

### 项目 1：计算与工具助手

基于 `basic_langgraph.py` 扩展。

功能：

- 数学计算
- 时间查询
- 单位换算
- 条件路由
- 工具调用日志
- 流式输出

适合练习：

- `ToolNode`
- `tool_calls`
- 条件边
- 基础调试

### 项目 2：本地文档问答助手

功能：

- 读取本地 Markdown / TXT 文件
- 简单检索相关片段
- 让模型基于片段回答
- 保存引用来源
- 增加 reviewer 节点检查答案是否脱离材料

适合练习：

- RAG
- 多字段 state
- 多节点工作流
- 答案审查

### 项目 3：任务执行 Agent

功能：

- planner 拆解任务
- executor 执行工具
- observer 观察结果
- replanner 修改计划
- reporter 输出最终报告

适合练习：

- 循环图
- 多步骤执行
- 错误恢复
- 复杂状态设计

## 9. 推荐学习节奏

### 第 1 周：基础图和状态

- 第 1 天：读懂 `basic_langgraph.py`
- 第 2 天：写最小图
- 第 3 天：练习 state 更新
- 第 4 天：练习 reducer
- 第 5 天：练习条件路由
- 第 6 天：复刻当前工具调用流程
- 第 7 天：整理笔记，画流程图

### 第 2 周：工具、循环和多节点

- 第 8 天：新增 2-3 个工具
- 第 9 天：打印工具调用和工具结果
- 第 10 天：拆分 planner / writer / reviewer
- 第 11 天：实现 reviewer 返工
- 第 12 天：把 `invoke` 改成 `stream`
- 第 13 天：做一个完整小项目
- 第 14 天：复盘 state 设计和图结构

### 第 3 周以后：工程化能力

- checkpoint
- thread_id
- human-in-the-loop
- time travel
- LangSmith tracing
- 测试和评估
- 部署结构

## 10. 开源综合示例和参考资料

下面这些资料按推荐阅读顺序排列。

### 1. LangGraph 官方文档

链接：https://docs.langchain.com/oss/python/langgraph

用途：

- 学最新 API
- 查概念说明
- 看官方推荐的能力边界
- 理解 durable execution、streaming、human-in-the-loop、memory 等核心能力

建议：

- 优先看 overview
- 然后看 quickstart
- 再看 graph API、functional API、streaming、persistence

### 2. LangGraph Quickstart

链接：https://docs.langchain.com/oss/python/langgraph/quickstart

用途：

- 对照 `basic_langgraph.py`
- 理解模型节点、工具节点、工具循环
- 学官方最小 agent 写法

注意：

- 官方 quickstart 现在更强调 Functional API
- 你的代码使用的是 Graph API，这很好，更适合先理解图结构

### 3. Thinking in LangGraph

链接：https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

用途：

- 建立 LangGraph 思维方式
- 学会如何把业务问题拆成 state、node、edge
- 从“写链”转向“设计状态机”

### 4. langchain-ai/langgraph

链接：https://github.com/langchain-ai/langgraph

用途：

- 看源码和 issue
- 了解版本变化
- 查 release
- 了解官方生态

注意：

- `examples/` 目录仍可参考，但官方说明该目录主要为归档用途，最新示例应以 LangChain 文档为准

### 5. LangGraph examples 归档目录

链接：https://github.com/langchain-ai/langgraph/tree/main/examples

用途：

- 参考历史 notebook
- 看 RAG、multi-agent、reflection、human-in-the-loop 等不同图模式

适合阶段：

- 学完基础 state、node、edge 后再看

### 6. NirDiamant/GenAI_Agents

链接：https://github.com/NirDiamant/GenAI_Agents

用途：

- 综合 Agent 案例库
- 包含 LangGraph、LangChain、PydanticAI 等不同框架示例
- 适合从项目角度理解 Agent 应用

建议：

- 不要一开始全看
- 先找里面的 `Introduction to LangGraph`
- 再看 customer support、travel planning、scientific paper agent 这类完整场景

### 7. LangGraph Academy

链接：https://academy.langchain.com

用途：

- 系统课程
- 适合在本地练习后查漏补缺

## 11. 建议避免的学习误区

1. 不要一开始就做复杂多 Agent
2. 不要只看 notebook，不自己画图
3. 不要把所有数据都塞进 `messages`
4. 不要把所有逻辑都写进一个节点
5. 不要忽略工具调用过程中的错误处理
6. 不要把 LangGraph 当成 LangChain agent 的替代黑盒
7. 不要在还没理解 state 前就急着学 checkpoint

## 12. 最小复盘问题

每学完一个阶段，可以用这些问题检查自己：

1. 当前图的 state 长什么样？
2. 每个节点读取哪些字段？
3. 每个节点更新哪些字段？
4. 哪些字段需要 reducer？
5. 条件边依赖哪个状态判断？
6. 图有没有循环？
7. 循环什么时候结束？
8. 如果节点失败，状态会停在哪里？
9. 如果要恢复执行，需要保存什么？
10. 如果要调试，应该打印哪个节点的输入输出？

## 13. 推荐主线

最稳的学习顺序是：

```text
basic_langgraph.py
  -> 最小图
  -> State 和 reducer
  -> 条件路由
  -> ToolNode
  -> 多节点 workflow
  -> stream
  -> checkpoint
  -> human-in-the-loop
  -> 综合项目
```

一句话总结：先把 LangGraph 当作“状态机”学，再把它当作“Agent 编排框架”用。

## 综合项目推荐：

1. 智能客服系统

- 多工具支持
- 会话管理
- 工具调用路由

2. 多 Agent 协作

- 规划 Agent（制定计划）
- 执行 Agent（执行工具）
- 总结 Agent（汇总结果）

3. 复杂工作流

- 数据处理流程
- 审批流程
- 多步骤任务
