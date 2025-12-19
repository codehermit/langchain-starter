import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from chat_models import init_chat_model
from datetime import datetime

load_dotenv()

model = init_chat_model("deepseek-v3.2", model_provider="openai", api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

@tool
def get_current_time() -> str:
    """Returns the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def simple_calculator(expression: str) -> str:
    """Returns the result of a simple calculator expression."""
    return str(eval(expression))

@tool
def mock_weather(city: str) -> str:
    """Returns the weather for a given city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model,
    tools=[get_current_time, simple_calculator, mock_weather],
    verbose=True,
)

question = "What is the weather in Beijing?"
print(f"👤 用户: {question}\n")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": question}
    ]
})

# 获取最后一条消息（AI 的回答）
answer = result["messages"][-1].content
print(f"🤖 Agent: {answer}\n")