import os
import importlib.util
import logging
import json
import time
import uuid
import contextvars
from functools import wraps
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM = True
except Exception:
    PROM = False
    Counter = None
    Histogram = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

load_dotenv()

app = FastAPI(title="Customer Service Agent API", version="1.0.0")

root = Path(__file__).resolve().parents[2]
project_dir = root / "project"
agent_file = project_dir / "05" / "cust_service_agent_cli.py"

spec = importlib.util.spec_from_file_location("cust_service_agent_cli", agent_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TRACE_ID = contextvars.ContextVar("trace_id", default="")
SESSION_ID = contextvars.ContextVar("session_id", default="")
logs_dir = project_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logs_dir / "app.jsonl", encoding="utf-8")
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

def count_tool_calls(result):
    cnt = 0
    for m in result.get("messages", []):
        t = getattr(m, "type", None)
        r = getattr(m, "role", None)
        c = getattr(m, "content", "")
        if t == "tool" or r == "tool":
            cnt += 1
        elif isinstance(c, str) and ("工具" in c or "Tool" in c):
            cnt += 1
    return cnt
agent = mod.create_customer_service_agent()

SESSION_MESSAGES: Dict[str, List[Dict[str, str]]] = {}
MAX_MESSAGES = 12
if PROM:
    REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["path", "method", "status"])
    REQUEST_LATENCY = Histogram("http_request_latency_seconds", "HTTP request latency", ["path", "method"])
    TOOL_CALLS = Counter("tool_calls_total", "Tool calls", ["tool", "status"])
else:
    REQUEST_COUNT = None
    REQUEST_LAT_SUM = {}
    REQUEST_LAT_COUNT = {}
    TOOL_CALLS = None
    SIMPLE_REQ_COUNT = {}
    SIMPLE_TOOL_CALLS = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    trace_id: str

@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    tid = str(uuid.uuid4())
    TRACE_ID.set(tid)
    sid = request.headers.get("X-Session-Id", "") or ""
    SESSION_ID.set(sid)
    resp = None
    status_code = 500
    try:
        resp = await call_next(request)
        status_code = resp.status_code
        return resp
    finally:
        elapsed = time.time() - t0
        if PROM:
            REQUEST_COUNT.labels(path=request.url.path, method=request.method, status=str(status_code)).inc()
            REQUEST_LATENCY.labels(path=request.url.path, method=request.method).observe(elapsed)
        else:
            SIMPLE_REQ_COUNT[(request.url.path, request.method, str(status_code))] = SIMPLE_REQ_COUNT.get((request.url.path, request.method, str(status_code)), 0) + 1
            key = (request.url.path, request.method)
            REQUEST_LAT_SUM[key] = REQUEST_LAT_SUM.get(key, 0.0) + elapsed
            REQUEST_LAT_COUNT[key] = REQUEST_LAT_COUNT.get(key, 0) + 1
        rec = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": "INFO",
            "trace_id": tid,
            "session_id": sid,
            "type": "access",
            "path": request.url.path,
            "method": request.method,
            "status": status_code,
            "latency_ms": int(elapsed * 1000),
        }
        logger.info(json.dumps(rec, ensure_ascii=False))


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/c_hello")
def c_hello():
    return {"c_hello": "hello, longchain"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or "default"
    SESSION_ID.set(sid)
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
    tool_calls = count_tool_calls(result)
    msgs.append({"role": "assistant", "content": answer})
    if len(msgs) > MAX_MESSAGES:
        SESSION_MESSAGES[sid] = msgs[-MAX_MESSAGES:]
    trace_id = TRACE_ID.get()
    logger.info(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level": "INFO",
        "trace_id": trace_id,
        "session_id": sid,
        "type": "chat",
        "user_message": req.message[:500],
        "reply_preview": answer[:500],
        "tool_calls_count": tool_calls,
        "reply_length": len(answer),
    }, ensure_ascii=False))
    return ChatResponse(reply=answer, session_id=sid, trace_id=trace_id)

@app.get("/metrics")
def metrics():
    if PROM:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    lines = []
    lines.append("# TYPE http_requests_total counter")
    for (path, method, status), cnt in SIMPLE_REQ_COUNT.items():
        lines.append(f'http_requests_total{{path="{path}",method="{method}",status="{status}"}} {cnt}')
    lines.append("# TYPE http_request_latency_seconds summary")
    for key, s in REQUEST_LAT_SUM.items():
        c = REQUEST_LAT_COUNT.get(key, 0)
        path, method = key
        lines.append(f'http_request_latency_seconds_sum{{path="{path}",method="{method}"}} {s}')
        lines.append(f'http_request_latency_seconds_count{{path="{path}",method="{method}"}} {c}')
    lines.append("# TYPE tool_calls_total counter")
    for (tool, status), cnt in SIMPLE_TOOL_CALLS.items():
        lines.append(f'tool_calls_total{{tool="{tool}",status="{status}"}} {cnt}')
    return Response(content="\n".join(lines) + "\n", media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
