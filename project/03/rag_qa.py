"""
阶段 2：RAG 本地知识库问答示例
功能：
1. 从 docs/faq.md 加载文档
2. 文本分块并建立向量库（FAISS）
3. 支持命令行问答，基于文档检索回答
4. 显示引用的文档片段

申明：rag_qa.py 里用的是 OpenAIEmbeddings + text-embedding-3-small，
但当前配置的deepseek-v3.2没有这个 embedding 模型，所以会报错。学者如果运行此文件，请配置一个支持embedding的模型。
"""

import os
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = DATA_DIR / "faiss_index"
FAQ_FILE = DOCS_DIR / "faq.md"


def init_model():
    """初始化聊天模型"""
    return init_chat_model(
        "deepseek-v3.2",
        model_provider="openai",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )


def init_embeddings():
    """初始化 Embedding 模型（用于向量化）"""
    # 使用 OpenAI 兼容的 Embedding API
    # 如果你的 API 支持 embedding，使用相同的 base_url
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # 或使用其他 embedding 模型
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL")
    )


def load_and_split_documents():
    """加载文档并分块"""
    print(f"正在加载文档: {FAQ_FILE}")
    
    if not FAQ_FILE.exists():
        raise FileNotFoundError(f"FAQ 文件不存在: {FAQ_FILE}")
    
    # 加载文档
    loader = TextLoader(str(FAQ_FILE), encoding="utf-8")
    documents = loader.load()
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每个 chunk 约 500 字符
        chunk_overlap=50,    # chunk 之间重叠 50 字符，避免信息丢失
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"文档已分块，共 {len(chunks)} 个片段")
    
    return chunks


def build_vector_store():
    """构建向量库并保存"""
    print("正在构建向量库...")
    
    # 加载并分块文档
    chunks = load_and_split_documents()
    
    # 初始化 embeddings
    embeddings = init_embeddings()
    
    # 创建向量库
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 保存向量库
    DATA_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    print(f"向量库已保存到: {VECTOR_STORE_PATH}")
    
    return vectorstore


def load_vector_store():
    """加载已存在的向量库"""
    print(f"正在加载向量库: {VECTOR_STORE_PATH}")
    
    embeddings = init_embeddings()
    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True  # FAISS 需要此参数
    )
    
    print("向量库加载成功")
    return vectorstore


def get_vector_store():
    """获取向量库（如果不存在则构建）"""
    if VECTOR_STORE_PATH.exists():
        try:
            return load_vector_store()
        except Exception as e:
            print(f"加载向量库失败: {e}")
            print("重新构建向量库...")
            return build_vector_store()
    else:
        return build_vector_store()


def create_rag_chain(model, retriever):
    """创建 RAG 链"""
    # 定义 prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能客服助手，基于提供的文档内容回答用户问题。
要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，诚实告知用户
3. 回答要简洁、准确、友好
4. 如果文档中有多个相关答案，可以综合回答

文档内容：
{context}

请基于以上文档内容回答用户问题。"""),
        ("user", "{question}")
    ])
    
    # 创建链：检索 -> 格式化 prompt -> 调用模型
    def rag_chain(question: str):
        # 1. 检索相关文档（新版 LangChain 中 retriever 是 Runnable，使用 invoke）
        docs = retriever.invoke(question)
        
        # 2. 将文档内容拼接成上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. 格式化 prompt
        messages = prompt.invoke({
            "context": context,
            "question": question
        })
        
        # 4. 调用模型
        response = model.invoke(messages)
        
        return response, docs
    
    return rag_chain


def main():
    """主函数"""
    print("=" * 60)
    print("RAG 本地知识库问答系统")
    print("=" * 60)
    
    # 初始化模型
    print("\n正在初始化模型...")
    model = init_model()
    
    # 获取向量库
    print("\n正在准备向量库...")
    vectorstore = get_vector_store()
    
    # 创建检索器（检索 top-3 相关文档）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 创建 RAG 链
    rag_chain = create_rag_chain(model, retriever)
    
    print("\n" + "=" * 60)
    print("系统就绪！请输入您的问题（输入 exit/quit/q 退出）")
    print("=" * 60 + "\n")
    
    # 交互式问答
    while True:
        question = input("您：")
        
        if question.lower() in ("exit", "quit", "q"):
            print("\n感谢使用，再见！")
            break
        
        if not question.strip():
            continue
        
        try:
            # 调用 RAG 链
            response, source_docs = rag_chain(question)
            
            # 显示回答
            print("\n助手：", response.content)
            
            # 显示引用的文档片段（可选）
            print("\n【参考文档片段】")
            for i, doc in enumerate(source_docs, 1):
                print(f"\n片段 {i}：")
                # 只显示前 200 字符，避免输出过长
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(preview)
            
            print("\n" + "-" * 60 + "\n")
            
        except Exception as e:
            print(f"\n错误：{e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

