"""
LangGraph 基础示例：最小版 langgraph_agent 框架

功能目标（对应学习文档第 9 阶段）：
- 定义 AgentState（包含 messages）
- 创建 call_model 节点：调用 LLM
- 创建 tool_node 节点：执行工具（使用 LangGraph 预置 ToolNode）
- 定义图结构：
  START -> call_model
  call_model -> (判断是否调用工具) -> tool_node / END
  tool_node -> call_model
- 提供一个简单的命令行交互 demo
"""

from __future__ import annotations

import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ===== 0. 初始化模型 =====

load_dotenv()

_model = init_chat_model(
    "deepseek-v3.2",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


# ===== 1. 定义 Agent 的状态 =====


class AgentState(TypedDict):
    """
    在 LangGraph 中，State 是在各个节点之间传递的数据结构。

    这里我们只维护对话消息列表 messages：
    - 使用 Annotated[List[AnyMessage], add_messages]：
      LangGraph 会自动把新返回的消息 append 到列表中，
      避免你在每个节点里手工拼接历史。
    """

    messages: Annotated[List[AnyMessage], add_messages]


# ===== 2. 定义 Tools（示例用一个简单计算器） =====


@tool
def simple_calculator(expression: str) -> str:
    """执行一个简单的 Python 算术表达式，例如：'1 + 2 * 3'。"""
    try:
        # 在真实项目中请务必使用安全的解析方式，这里仅为 Demo
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果为：{result}"
    except Exception as e:
        return f"计算失败：{e}"


TOOLS = [simple_calculator]
tool_node = ToolNode(TOOLS)


# ===== 3. 定义调用模型的节点 =====


def call_model(state: AgentState) -> AgentState:
    """
    调用大模型的节点。

    - 输入：当前 state（包含历史 messages）
    - 输出：新增一条 AI 消息（LangGraph 会自动 append 到 messages 中）
    """
    # 绑定工具，使得模型可以在思考时“提出调用工具的请求”
    model_with_tools = _model.bind_tools(TOOLS)
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# ===== 4. 定义路由逻辑：是否需要调用工具 =====


def should_continue(state: AgentState) -> str:
    """
    根据当前最后一条 AI 消息，判断下一步走向：
    - 如果 AI 消息中包含 tool_calls，则进入 tools 节点
    - 否则说明可以直接给出最终回答，结束对话
    """
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]

    # langchain_core.messages.AIMessage 上通常有 tool_calls 属性
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "end"


# ===== 5. 构建 Graph =====


def build_graph():
    """构建并编译 LangGraph 图，返回一个可调用对象 graph_app。"""
    graph = StateGraph(AgentState)

    # 节点
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)

    # 入口：START -> call_model
    graph.set_entry_point("call_model")

    # 从 call_model 出发，根据 should_continue 的返回值进行路由
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # 工具调用完成后，回到 call_model 继续思考
    graph.add_edge("tools", "call_model")

    return graph.compile()


graph_app = build_graph()


# ===== 6. 简单命令行 Demo =====


def main() -> None:
    """
    命令行交互示例：
    - 你可以问一般性问题
    - 也可以让模型自己决定是否调用 simple_calculator 工具
      例如输入：帮我算一下 1314 * 520
    """
    print("=" * 60)
    print("LangGraph Agent Demo（基础版）")
    print("=" * 60)
    print("提示：输入 exit / quit / q 退出对话。\n")

    # 初始化对话状态：加入一个 system 提示词
    state: AgentState = {
        "messages": [
            SystemMessage(
                content=(
                    "你是一个简单的中文智能助手，可以在需要时调用 simple_calculator 工具，"
                    "帮助用户进行基础算术运算。"
                )
            )
        ]
    }

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("对话结束，再见！")
            break
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        # 运行图：invoke 会从 START 一直跑到 END
        state = graph_app.invoke(state)

        # 找出最后一条 AI 消息并打印
        last_ai = None
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "ai":
                last_ai = msg
                break

        if last_ai is not None:
            print(f"助手：{last_ai.content}\n")
        else:
            print("助手：暂无回复（未找到 AI 消息）\n")


if __name__ == "__main__":
    main()


