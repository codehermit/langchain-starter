"""
阶段 4：RAG + Agent 打造"客服 Agent 原型"
功能：
1. 将 RAG 问答封装成一个 Tool（faq_rag_tool）
2. 增加业务工具（mock 数据）：
   - query_order_status：查询订单状态
   - query_shipping_info：查询物流信息
3. 使用 Agent 组合这些工具，让 Agent 自动决定使用哪个工具
4. 支持命令行交互
"""

import os
from pathlib import Path
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = DATA_DIR / "faiss_index_local"

# 全局变量，用于缓存模型和检索器
_model = None
_retriever = None
_session_histories = {}
_MAX_MESSAGES = 12


def init_model():
    """初始化聊天模型"""
    global _model
    if _model is None:
        _model = init_chat_model(
            "deepseek-v4-flash",
            model_provider="openai",
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
    return _model


def init_embeddings():
    """初始化本地 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    history = _session_histories.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        _session_histories[session_id] = history
    if len(history.messages) > _MAX_MESSAGES:
        history.messages = history.messages[-_MAX_MESSAGES:]
    return history


def get_retriever():
    """获取或创建检索器"""
    global _retriever
    if _retriever is None:
        print("正在加载向量库...")
        embeddings = init_embeddings()
        vectorstore = FAISS.load_local(
            str(VECTOR_STORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        # 创建检索器（检索 top-3 相关文档）
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("向量库加载成功")
    return _retriever


@tool
def faq_rag_tool(question: str) -> str:
    """
    基于 FAQ 知识库回答用户问题。适用于：
    - 退款退货相关问题
    - 订单流程相关问题
    - 物流配送相关问题
    - 支付相关问题
    - 账户与会员相关问题
    - 售后服务相关问题
    
    Args:
        question: 用户的问题
        
    Returns:
        基于 FAQ 文档的回答
    """
    try:
        # 获取检索器和模型
        retriever = get_retriever()
        model = init_model()
        
        # 检索相关文档
        docs = retriever.invoke(question)
        
        # 将文档内容拼接成上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 创建 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能客服助手，基于提供的 FAQ 文档内容回答用户问题。
要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，诚实告知用户
3. 回答要简洁、准确、友好
4. 如果文档中有多个相关答案，可以综合回答

FAQ 文档内容：
{context}

请基于以上文档内容回答用户问题。"""),
            ("user", "{question}")
        ])
        
        # 格式化 prompt 并调用模型
        messages = prompt.invoke({
            "context": context,
            "question": question
        })
        
        response = model.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"查询 FAQ 时发生错误：{str(e)}"


@tool
def query_order_status(order_id: str) -> str:
    """
    查询订单状态。根据订单号返回订单的当前状态信息。
    
    Args:
        order_id: 订单号（例如：123456）
        
    Returns:
        订单状态的详细信息
    """
    # Mock 数据：模拟不同订单号返回不同状态
    mock_orders = {
        "123456": {
            "status": "已发货",
            "order_time": "2024-01-15 10:30:00",
            "total_amount": "299.00",
            "items": ["商品A x1", "商品B x2"]
        },
        "123457": {
            "status": "待发货",
            "order_time": "2024-01-20 14:20:00",
            "total_amount": "599.00",
            "items": ["商品C x1"]
        },
        "123458": {
            "status": "已完成",
            "order_time": "2024-01-10 09:15:00",
            "total_amount": "199.00",
            "items": ["商品D x1"]
        }
    }
    
    # 规范化订单号
    oid_raw = order_id or ""
    oid_digits = re.sub(r"\D", "", oid_raw)
    resolved_id = None
    if oid_digits in mock_orders:
        resolved_id = oid_digits
    else:
        for k in mock_orders.keys():
            if k in oid_raw:
                resolved_id = k
                break
    # 返回对应信息或默认
    if resolved_id in mock_orders:
        order = mock_orders[resolved_id]
        return f"""订单号：{order_id}
订单状态：{order['status']}
下单时间：{order['order_time']}
订单金额：¥{order['total_amount']}
商品信息：{', '.join(order['items'])}"""
    else:
        # 默认返回一个通用状态
        return f"""订单号：{order_id}
订单状态：待付款
下单时间：2024-01-25 16:00:00
订单金额：¥0.00
提示：如果订单号不正确，请检查后重试。"""


@tool
def query_shipping_info(order_id: str) -> str:
    """
    查询物流信息。根据订单号返回物流配送的详细信息。
    
    Args:
        order_id: 订单号（例如：123456）
        
    Returns:
        物流信息的详细信息，包括快递公司、快递单号、物流轨迹等
    """
    # Mock 数据：模拟不同订单号的物流信息
    mock_shipping = {
        "123456": {
            "carrier": "顺丰快递",
            "tracking_number": "SF1234567890123",
            "status": "运输中",
            "current_location": "北京分拨中心",
            "estimated_delivery": "2024-01-22",
            "tracking": [
                {"time": "2024-01-18 10:00", "location": "商家已发货", "status": "已揽收"},
                {"time": "2024-01-18 15:30", "location": "北京分拨中心", "status": "运输中"},
            ]
        },
        "123457": {
            "carrier": "中通快递",
            "tracking_number": "ZTO9876543210987",
            "status": "待发货",
            "current_location": "商家仓库",
            "estimated_delivery": "预计 1-3 个工作日发货",
            "tracking": [
                {"time": "2024-01-20 14:20", "location": "订单已确认", "status": "待发货"},
            ]
        },
        "123458": {
            "carrier": "圆通快递",
            "tracking_number": "YTO4567890123456",
            "status": "已签收",
            "current_location": "已送达",
            "estimated_delivery": "2024-01-12（已送达）",
            "tracking": [
                {"time": "2024-01-10 09:15", "location": "商家已发货", "status": "已揽收"},
                {"time": "2024-01-11 12:00", "location": "上海分拨中心", "status": "运输中"},
                {"time": "2024-01-12 14:30", "location": "上海XX区", "status": "派送中"},
                {"time": "2024-01-12 16:00", "location": "已签收", "status": "已签收"},
            ]
        }
    }
    
    # 规范化订单号或快递单号，尽可能解析到订单号
    oid_raw = order_id or ""
    oid_digits = re.sub(r"\D", "", oid_raw)
    resolved_id = None
    if oid_digits in mock_shipping:
        resolved_id = oid_digits
    else:
        for k in mock_shipping.keys():
            if k in oid_raw:
                resolved_id = k
                break
        if resolved_id is None:
            for k, v in mock_shipping.items():
                tn = v.get("tracking_number", "")
                tn_digits = re.sub(r"\D", "", tn)
                if oid_raw and oid_raw in tn:
                    resolved_id = k
                    break
                if oid_digits and oid_digits == tn_digits:
                    resolved_id = k
                    break
    # 返回对应信息或默认
    if resolved_id in mock_shipping:
        shipping = mock_shipping[resolved_id]
        result = f"""订单号：{order_id}
快递公司：{shipping['carrier']}
快递单号：{shipping['tracking_number']}
物流状态：{shipping['status']}
当前位置：{shipping['current_location']}
预计送达：{shipping['estimated_delivery']}

物流轨迹："""
        for track in shipping['tracking']:
            result += f"\n  {track['time']} - {track['location']} ({track['status']})"
        return result
    else:
        # 默认返回一个通用信息
        return f"""订单号：{order_id}
物流状态：暂无物流信息
提示：如果订单尚未发货或订单号不正确，将无法查询到物流信息。请确认订单号是否正确，或联系客服咨询。"""


def create_customer_service_agent():
    """创建客服 Agent"""
    model = init_model()
    
    # 定义所有工具
    tools = [faq_rag_tool, query_order_status, query_shipping_info]
    
    # 创建 Agent
    agent = create_agent(
        model,
        tools=tools,
        debug=True,  # 开启调试模式，可以看到 Agent 的思考过程
        system_prompt="""你是一个专业的智能客服助手，能够帮助用户解决各种问题。

你的能力包括：
1. 回答常见问题（FAQ）：使用 faq_rag_tool 工具查询知识库，回答关于退款、退货、订单、物流、支付、账户等问题
2. 查询订单状态：使用 query_order_status 工具查询订单的当前状态
3. 查询物流信息：使用 query_shipping_info 工具查询订单的物流配送情况

工作原则：
- 根据用户问题，智能选择合适的工具或工具组合
- 如果用户询问常见问题（如"如何申请退款"、"配送范围"等），使用 faq_rag_tool
- 如果用户询问具体订单的状态，使用 query_order_status（需要从用户输入中提取订单号）
- 如果用户询问具体订单的物流情况，使用 query_shipping_info（需要从用户输入中提取订单号）
- 回答要友好、专业、准确
- 在回答中简要说明你使用了什么工具来帮助用户（例如："我查询了您的订单信息..."）

对话记忆：
- 记住用户昵称与偏好（如语气、简洁程度），在后续回复中保持一致
- 记住当前会话中最近提到的订单号作为“上下文订单号”
- 当用户未明确给出订单号时，默认使用上下文订单号进行查询
- 如上下文中不存在订单号，礼貌地引导用户提供

现在开始为用户提供帮助吧！"""
    )
    
    return agent


def main():
    """主函数：命令行交互"""
    print("=" * 60)
    print("智能客服 Agent 系统")
    print("=" * 60)
    print("\n正在初始化系统...")
    
    # 创建 Agent
    agent = create_customer_service_agent()
    session_id = "cli"
    session_messages = []
    
    print("\n" + "=" * 60)
    print("系统就绪！我是您的智能客服助手，可以帮您：")
    print("  - 回答常见问题（退款、退货、订单、物流等）")
    print("  - 查询订单状态")
    print("  - 查询物流信息")
    print("\n请输入您的问题（输入 exit/quit/q 退出）")
    print("=" * 60 + "\n")
    
    # 交互式对话
    while True:
        try:
            user_input = input("👤 您：").strip()
            
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n感谢使用，再见！")
                break
            
            if not user_input:
                continue
            
            session_messages.append({"role": "user", "content": user_input})
            if len(session_messages) > _MAX_MESSAGES:
                session_messages = session_messages[-_MAX_MESSAGES:]
            result = agent.invoke({"messages": session_messages})
            
            # 获取最后一条消息（AI 的回答）
            answer = result["messages"][-1].content
            print(f"\n🤖 客服：{answer}\n")
            print("-" * 60 + "\n")
            session_messages.append({"role": "assistant", "content": answer})
            if len(session_messages) > _MAX_MESSAGES:
                session_messages = session_messages[-_MAX_MESSAGES:]
            
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

