import os
import importlib.util
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Customer Service Agent API", version="1.0.0")

root = Path(__file__).resolve().parents[2]
project_dir = root / "project"
agent_file = project_dir / "05" / "cust_service_agent_cli.py"

spec = importlib.util.spec_from_file_location("cust_service_agent_cli", agent_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
agent = mod.create_customer_service_agent()

SESSION_MESSAGES: Dict[str, List[Dict[str, str]]] = {}
MAX_MESSAGES = 12


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    trace_id: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/c_hello")
def c_hello():
    return {"c_hello": "hello, longchain"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or "default"
    msgs = SESSION_MESSAGES.get(sid)
    if msgs is None:
        msgs = []
        SESSION_MESSAGES[sid] = msgs
    msgs.append({"role": "user", "content": req.message})
    if len(msgs) > MAX_MESSAGES:
        SESSION_MESSAGES[sid] = msgs[-MAX_MESSAGES:]
        msgs = SESSION_MESSAGES[sid]
    result = agent.invoke({"messages": msgs})
    answer = result["messages"][-1].content
    msgs.append({"role": "assistant", "content": answer})
    if len(msgs) > MAX_MESSAGES:
        SESSION_MESSAGES[sid] = msgs[-MAX_MESSAGES:]
    trace_id = os.getenv("TRACE_ID", "")
    return ChatResponse(reply=answer, session_id=sid, trace_id=trace_id)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
