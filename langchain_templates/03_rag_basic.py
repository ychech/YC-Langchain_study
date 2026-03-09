#!/usr/bin/env python3
"""
LangChain 1.0.x 快速上手 - 基础 RAG (检索增强生成)
功能：加载文档、构建向量库、基于文档回答问题

注意：LangChain 1.0 中 RAG 相关功能已移到 langchain-classic 包
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta/v1",
    temperature=0.7,
    max_tokens=2048
)

# ========== 步骤1：准备文档 ==========
doc_content = """
LangChain 是一个用于开发大语言模型应用程序的框架123。
它提供了以下核心功能：
1. 模型调用：支持多种 LLM 和聊天模型
2. 提示词管理：模板化提示词，支持少样本学习
3. 链式调用：将多个组件组合成复杂工作流
4. 数据增强检索：RAG 实现，结合外部知识
5. 智能代理：让模型自主决策使用工具
6. 记忆管理：支持短期和长期记忆

RAG (Retrieval-Augmented Generation) 是 LangChain 的重要特性，
它通过检索相关文档来增强模型的回答能力，减少幻觉。

LangChain 1.0 于 2025年10月发布，是首个稳定版本，
承诺在 2.0 之前不会有破坏性变更。
"""

# 直接创建文档对象
# 这时 metadata["source"] 就会记录真实的来源（文件名、URL 等），方便后续追溯信息的出处
documents = [Document(page_content=doc_content, metadata={"source": "intro.txt"})]
print(f"✅ 加载了 {len(documents)} 个文档")

# ========== 步骤2：分割文档 ==========
# LangChain 1.0 中 text_splitter 在 langchain-text-splitters 包
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每个文本块的最大字符数
    chunk_overlap=50,      # 相邻文本块之间的重叠字符数，保持了语义连贯性
    separators=["\n\n", "\n", "。", ",", " ", ""]  # 分割优先级符号
)
chunks = text_splitter.split_documents(documents)

print(f"✅ 分割成 {len(chunks)} 个文档块")

# ========== 步骤3：构建向量库 ==========
print("🔄 正在加载嵌入模型（首次运行需要下载，请稍候）...")

# 使用 HuggingFace（人工智能平台） 本地嵌入（Embeddings）
from langchain_huggingface import HuggingFaceEmbeddings
#~/.cache/huggingface/hub/这里模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True} # 向量归一化
)
print("✅ 嵌入模型加载完成")

# 构建 FAISS 向量库（在 langchain-classic 中）
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ 向量库构建完成")

#在向量库中检索（召回）最相似的文档块
#"检索" 更通用，指整个查找过程
#"召回" 强调从海量数据中"捞取"相关内容的动作
# 创建检索器，用于从向量库中搜索与问题最相关的文档块
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 使用相似度搜索，余弦相似度算法进行向量检索
    search_kwargs={"k": 3}     # 召回 top 3 个最相关的文档，每次检索返回最相似的 3 个文档块作为上下文
)


# ========== 步骤4：构建 RAG 链 ==========
# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是专业的AI助手。请基于以下检索到的上下文回答问题。
如果上下文中没有相关信息，请明确说明。

上下文：
{context}
"""),
    ("user", "{input}")
])

# 这是一个文档格式化函数，用于将检索到的多个文档块整理成统一的文本格式。
def format_docs(docs):
    return "\n\n".join([f"[来自 {doc.metadata.get('source', '未知')}]: {doc.page_content}" for doc in docs])

# 构建 RAG 链（LCEL 风格）
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ========== 步骤5：测试问答 ==========
print("\n===== RAG 问答测试 =====\n")

questions = [
    "LangChain 是什么？",
    "RAG 有什么作用？",
    "LangChain 1.0 什么时候发布的？",
]

for question in questions:
    print(f"问题：{question}")
    
    # 先查看检索到的原始文档（带来源标签）
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    print(f"\n📄 检索到的原始上下文:")
    print("=" * 60)
    print(formatted_context)
    print("=" * 60)

    #正式回答
    response = rag_chain.invoke(question)
    print(f"正式回答：{response.content}\n")
    print("-" * 50)

print("\n✅ RAG 测试完成")
