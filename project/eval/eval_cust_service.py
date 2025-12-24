import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import importlib.util

def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def evaluate_case(agent, case, session_messages):
    history = case.get("history", [])
    messages = case.get("messages", [])
    merged = session_messages + history + messages
    result = agent.invoke({"messages": merged})
    answer = result["messages"][-1].content
    for m in messages:
        session_messages.append(m)
    session_messages.append({"role": "assistant", "content": answer})
    return answer

def keyword_hits(text, keywords):
    hits = []
    for k in keywords:
        if k in text:
            hits.append(k)
    return hits

def main():
    # 加载环境变量（例如 API_KEY、BASE_URL），确保被被测 Agent 初始化时可用
    load_dotenv()
    # 计算仓库根目录：当前文件在 project/eval/ 目录下，向上两级即为仓库根
    root = Path(__file__).resolve().parents[2]
    # 项目目录路径（用于定位被测 Agent 代码与输出目录）
    project_dir = root / "project"
    # 被测 Agent 的 Python 文件路径（直接定位到 CLI 脚本，动态导入其中的构建函数）
    agent_file = project_dir / "05" / "cust_service_agent_cli.py"
    # 测试用例文件路径（JSON 结构，含多条用例）
    cases_path = root / "tests" / "cases.json"
    # 评估输出目录（存放 JSONL 结果，便于后续分析）
    out_dir = project_dir / "eval" / "output"
    ensure_dir(out_dir)
    # 输出文件名包含时间戳，避免覆盖历史
    out_file = out_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    # 通过 importlib 动态加载被测 Agent 模块，避免对项目结构的硬编码依赖
    spec = importlib.util.spec_from_file_location("cust_service_agent_cli", agent_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 调用模块内的工厂函数，创建一个可交互的客服 Agent 实例
    agent = mod.create_customer_service_agent()
    # 加载测试用例列表
    cases = load_cases(cases_path)
    # 单个评估会话的消息历史（模拟连续对话场景；如需独立会话可为每条用例创建新列表）
    session_messages = []
    # 汇总结果（用于计算简单统计）
    results = []
    # 逐条用例执行评估：调用 Agent → 收集回答 → 进行关键词命中统计 → 写出 JSONL
    with out_file.open("w", encoding="utf-8") as wf:
        for case in cases:
            cid = case.get("id", "")
            answer = evaluate_case(agent, case, session_messages)
            keys = case.get("expect_keywords", [])
            hits = keyword_hits(answer, keys)
            record = {
                "id": cid,
                "answer_length": len(answer),
                "expect_keywords": keys,
                "hit_keywords": hits,
                "hit_count": len(hits),
                "hit_rate": (len(hits) / len(keys)) if keys else 0.0,
                "answer": answer
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
    # 计算并打印简单统计（用例数、平均回答长度、平均关键词命中率），便于快速评估质量
    total = len(results)
    avg_len = sum(r["answer_length"] for r in results) / total if total else 0
    avg_hit = sum(r["hit_rate"] for r in results) / total if total else 0.0
    summary = {
        "total_cases": total,
        "avg_answer_length": avg_len,
        "avg_hit_rate": avg_hit
    }
    print("评估完成")
    print(f"用例数: {summary['total_cases']}")
    print(f"平均回答长度: {summary['avg_answer_length']:.1f}")
    print(f"平均命中率: {summary['avg_hit_rate']:.2f}")
    print(f"结果文件: {out_file}")

if __name__ == "__main__":
    main()
