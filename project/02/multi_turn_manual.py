import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "deepseek-v3.2",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 使用 MessagesPlaceholder 来放置历史对话
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个耐心的中文讲解助手，会用简洁易懂的方式回答问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{user_input}")
])

if __name__ == "__main__":
    # 手工维护的历史列表，存储 [(role, content), ...]
    history = []
    
    print("多轮对话（手工管理历史），输入 exit/quit/q 退出")
    
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ("exit", "quit", "q"):
            print("对话结束。")
            break
        
        # 手工把历史拼接到 prompt 中传给模型
        # 体会：上下文都是通过 prompt 带给模型的
        messages = prompt.invoke({
            "history": history,
            "user_input": user_input
        })
        
        response = model.invoke(messages)
        print("助手：", response.content)
        
        # 手工将本轮对话加入历史
        history.append(("user", user_input))
        history.append(("assistant", response.content))
