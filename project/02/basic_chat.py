import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "deepseek-v3.2",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 使用 ChatPromptTemplate 组合系统角色和用户输入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个耐心的中文讲解助手，会用简洁易懂的方式回答问题。"),
    ("user", "{user_input}")
])

if __name__ == "__main__":
    user_input = input("请输入你的问题：")
    
    # 格式化 prompt 并调用模型
    messages = prompt.invoke({"user_input": user_input})
    response = model.invoke(messages)
    
    print("\n助手：", response.content)
