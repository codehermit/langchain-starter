# 阶段 2：RAG 本地知识库问答

本目录包含 RAG（检索增强生成）的示例代码。

## 文件说明

- **`rag_qa.py`**：使用 API Embedding 的版本（如果你的 API 支持 embedding）
- **`rag_qa_local_embedding.py`**：使用本地 HuggingFace Embedding 模型的版本（推荐，更稳定）

## 使用方法

### 方式一：使用本地 Embedding（推荐）

```bash
# 安装依赖
pip install -r requirements.txt

# 运行（首次运行会下载 embedding 模型，需要几分钟）
python project/03/rag_qa_local_embedding.py
```

### 方式二：使用 API Embedding

如果你的 API 支持 embedding（如 OpenAI、一些兼容 OpenAI 的 API），可以使用：

```bash
python project/03/rag_qa.py
```

**注意**：DeepSeek API 可能不支持 embedding，如果遇到错误，请使用方式一。

## 功能说明

1. **首次运行**：
   - 从 `docs/faq.md` 加载文档
   - 将文档分块
   - 使用 embedding 模型向量化
   - 保存向量库到 `data/faiss_index_local/`（或 `data/faiss_index/`）

2. **后续运行**：
   - 直接加载已保存的向量库，无需重新构建

3. **问答功能**：
   - 输入问题，系统会：
     - 检索最相关的文档片段（top-3）
     - 将文档片段和问题一起传给模型
     - 生成基于文档的答案
     - 显示引用的文档片段

## 示例问题

- "如何申请退款？"
- "订单多久发货？"
- "支持哪些支付方式？"
- "如何查询物流信息？"

## 目录结构

```
project/
├── 03/
│   ├── rag_qa.py                    # API Embedding 版本
│   ├── rag_qa_local_embedding.py    # 本地 Embedding 版本（推荐）
│   └── README.md                    # 本文件
├── docs/
│   └── faq.md                       # FAQ 文档
└── data/
    └── faiss_index_local/           # 向量库（自动生成）
```

## 常见问题

### Q: 首次运行很慢？
A: 首次运行需要下载 embedding 模型（约几百MB），这是正常的。后续运行会直接使用本地模型。

### Q: 如何更新知识库？
A: 修改 `docs/faq.md` 后，删除 `data/faiss_index_local/` 目录，重新运行脚本即可重建向量库。

### Q: 如何调整检索的文档数量？
A: 修改代码中的 `search_kwargs={"k": 3}`，将 3 改为你想要的数量。

### Q: 如何支持更多文档格式？
A: LangChain 支持多种文档加载器，可以修改 `load_and_split_documents()` 函数，使用 `PyPDFLoader`、`UnstructuredFileLoader` 等。

