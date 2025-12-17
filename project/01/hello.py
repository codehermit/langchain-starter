import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# print(os.getenv("API_KEY"))
# print(os.getenv("BASE_URL"))

model = init_chat_model(
    "deepseek-v3.2",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 非流式输出
# response = model.invoke("用三句话介绍大模型?")
# print(response)

# 流式输出
# for chunk in model.stream("用三句话介绍大模型?"):
#     print(chunk.content, end="", flush=True)

# 批量输出
messages = [
    ("user", "用三句话介绍大模型?"),
    ("assistant", "大模型是一种人工智能模型，它能够理解自然语言，并生成自然语言。"),
    ("user", "大模型有哪些特点？"),
    ("assistant", "大模型具有以下特点："),
    ("user", "大模型有哪些应用？"),
    ("assistant", "大模型可以应用于以下领域："),
]
# response = model.batch(messages)
# print(response)

# # 批量流式输出
# for chunk in model.stream(messages):
#     print(chunk.content, end="", flush=True)
# 批量流式输出这部分代码不对，实际上只能输出最后一个问题的回答，而不是所有问题的回答

# 原因：stream 只针对「一次输入」，你的 messages 被当成一整段对话，在 LangChain 里，
# messages 会被解释为 一段完整的聊天历史（一次输入），而不是「三条问题的 batch」。
# model.stream(messages) 的含义是：基于这整段历史，生成“下一条 assistant 回复”并流式输出，而“下一条回复”自然只会针对「最后一个 user 消息」（也就是“大模型有哪些应用？”）。
# 所以你只看到最后一个问题对应的回答，这是符合 stream 语义的。

# 解决方法一：使用batch输出
questions = [
    "用三句话介绍大模型?",
    "大模型有哪些特点？",
    "大模型有哪些应用？",
]

# responses = model.batch(questions)
# for q, r in zip(questions, responses):
#     print("\n==== 问题 ====")
#     print(q)
#     print("==== 回答 ====")
#     print(r.content)

# 解决方法二：多条问题 + 流式输出——对每个问题单独 stream
# for q in questions:
#     print(f"\n\n==== 问题 ====\n{q}\n==== 回答（流式） ====")
#     for chunk in model.stream(q):
#         print(chunk.content, end="", flush=True)

# 多轮对话示例：使用一个 history 列表保存上下文
# 每一轮把完整 history 传给 model.invoke，这样模型就能“记住”前面的对话
# 在命令行中输入内容，输入 exit / quit / q 退出对话
if __name__ == "__main__":
    history = []

    while True:
        user_input = input("\n你：")
        if user_input.lower() in ("exit", "quit", "q"):
            print("对话结束。")
            break

        # 1. 先把用户这轮话加入历史
        history.append(("user", user_input))

        # 2. 调用模型，传入完整历史
        response = model.invoke(history)
        print("助手：", response.content)

        # 3. 再把模型回答加入历史，供下一轮使用
        history.append(("assistant", response.content))
