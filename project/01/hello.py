import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("API_KEY"))
print(os.getenv("BASE_URL"))

model = init_chat_model(
    "deepseek-v3.2",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

response = model.invoke("用三句话介绍大模型?")
print(response)