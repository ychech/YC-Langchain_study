#!/usr/bin/env python3
"""
LangChain Community 核心概念 06 - 检索器 (Retrievers)
功能：学习各种检索策略和检索器

核心概念：
- BaseRetriever: 检索器基类
- VectorStoreRetriever: 向量检索
- MultiQueryRetriever: 多查询检索
- ContextualCompression: 上下文压缩
"""
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import (
    MultiQueryRetriever,
    BM25Retriever
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def get_sample_documents():
    """获取示例文档"""
    return [
        Document(page_content="Python 是一种高级编程语言，简洁优雅，适合初学者", metadata={"source": "doc1", "category": "programming"}),
        Document(page_content="JavaScript 是网页开发的主要语言，运行在浏览器中", metadata={"source": "doc2", "category": "programming"}),
        Document(page_content="Java 是一种面向对象的编程语言，广泛用于企业开发", metadata={"source": "doc3", "category": "programming"}),
        Document(page_content="机器学习是人工智能的一个分支，让计算机从数据中学习", metadata={"source": "doc4", "category": "ai"}),
        Document(page_content="深度学习使用神经网络进行学习，是机器学习的一种方法", metadata={"source": "doc5", "category": "ai"}),
        Document(page_content="自然语言处理让计算机理解和生成人类语言", metadata={"source": "doc6", "category": "ai"}),
        Document(page_content="Python 在数据科学和机器学习领域非常流行", metadata={"source": "doc7", "category": "data_science"}),
        Document(page_content="数据分析使用 Python 的 pandas 和 numpy 库", metadata={"source": "doc8", "category": "data_science"}),
    ]


def demo_basic_retriever():
    """演示基础检索器"""
    print("=" * 60)
    print("LangChain Community 核心概念 06 - 检索器")
    print("=" * 60)
    
    print("\n🔍 基础向量检索器\n")
    print("-" * 50)
    
    # 准备数据
    docs = get_sample_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 创建向量库
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    print("\n1️⃣ 基础检索:")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    query = "Python 编程"
    results = retriever.invoke(query)
    
    print(f"   查询: '{query}'")
    print(f"   返回 {len(results)} 个结果:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      分类: {doc.metadata['category']}\n")


def demo_search_types():
    """演示不同搜索类型"""
    print("\n" + "=" * 60)
    print("检索类型对比")
    print("=" * 60)
    
    docs = get_sample_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    query = "人工智能"
    
    print(f"\n📊 查询: '{query}'\n")
    
    # 1. 相似度搜索
    print("1️⃣ Similarity Search (相似度搜索):")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    results = retriever.invoke(query)
    for doc in results:
        print(f"   - {doc.page_content[:50]}...")
    
    # 2. MMR 搜索
    print("\n2️⃣ MMR Search (最大边际相关性):")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5}
    )
    results = retriever.invoke(query)
    for doc in results:
        print(f"   - {doc.page_content[:50]}...")
    
    # 3. 相似度阈值搜索
    print("\n3️⃣ Similarity Score Threshold (相似度阈值):")
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )
    results = retriever.invoke(query)
    print(f"   找到 {len(results)} 个结果 (阈值 0.5)")


def demo_bm25_retriever():
    """演示 BM25 检索器"""
    print("\n" + "=" * 60)
    print("BM25 Retriever - 基于关键词的检索")
    print("=" * 60)
    
    print("\n📚 BM25 是一种经典的信息检索算法\n")
    
    docs = get_sample_documents()
    
    # 创建 BM25 检索器
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 3  # 设置返回数量
    
    query = "Python 机器学习"
    results = retriever.invoke(query)
    
    print(f"   查询: '{query}'")
    print(f"   返回 {len(results)} 个结果:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")


def demo_multi_query_retriever():
    """演示多查询检索器"""
    print("\n" + "=" * 60)
    print("MultiQueryRetriever - 多查询检索")
    print("=" * 60)
    
    print("\n🔄 从多个角度生成查询，提高召回率\n")
    
    from langchain_openai import ChatOpenAI
    
    docs = get_sample_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 基础检索器
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 创建多查询检索器
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    query = "编程语言"
    print(f"   原始查询: '{query}'")
    print(f"   MultiQueryRetriever 会生成多个相关查询...\n")
    
    results = retriever.invoke(query)
    
    print(f"   返回 {len(results)} 个结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")


def demo_custom_retriever():
    """演示自定义检索器"""
    print("\n" + "=" * 60)
    print("自定义检索器")
    print("=" * 60)
    
    print("\n🏗️ 继承 BaseRetriever 创建自定义检索器:\n")
    
    class CategoryFilterRetriever(BaseRetriever):
        """按分类过滤的检索器"""
        
        def __init__(self, vectorstore, category: str = None, **kwargs):
            super().__init__(**kwargs)
            self.vectorstore = vectorstore
            self.category = category
        
        def _get_relevant_documents(self, query: str) -> List[Document]:
            """获取相关文档"""
            # 先进行向量检索
            docs = self.vectorstore.similarity_search(query, k=10)
            
            # 按分类过滤
            if self.category:
                docs = [d for d in docs if d.metadata.get("category") == self.category]
            
            return docs[:3]  # 返回前3个
    
    # 使用自定义检索器
    docs = get_sample_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    retriever = CategoryFilterRetriever(vectorstore, category="ai")
    
    query = "学习"
    results = retriever.invoke(query)
    
    print(f"   查询: '{query}'")
    print(f"   过滤条件: category='ai'")
    print(f"   返回 {len(results)} 个结果:\n")
    
    for doc in results:
        print(f"   - {doc.page_content}")
        print(f"     分类: {doc.metadata['category']}")


def show_retriever_comparison():
    """检索器对比"""
    print("\n" + "=" * 60)
    print("检索器对比")
    print("=" * 60)
    
    print("""
┌─────────────────────────┬─────────────────────────────┬──────────────────┐
│ 检索器                   │ 特点                        │ 适用场景         │
├─────────────────────────┼─────────────────────────────┼──────────────────┤
│ VectorStoreRetriever    │ 基于语义相似度               │ 语义搜索         │
│ BM25Retriever          │ 基于关键词匹配               │ 精确匹配         │
│ MultiQueryRetriever    │ 生成多查询提高召回           │ 查询理解困难     │
│ EnsembleRetriever      │ 组合多个检索器               │ 综合检索         │
│ ParentDocumentRetriever│ 检索父文档                   │ 需要完整上下文   │
│ SelfQueryRetriever     │ 从查询中提取过滤条件         │ 结构化数据       │
└─────────────────────────┴─────────────────────────────┴──────────────────┘
    """)


def main():
    demo_basic_retriever()
    demo_search_types()
    demo_bm25_retriever()
    demo_multi_query_retriever()
    demo_custom_retriever()
    show_retriever_comparison()
    
    print("\n" + "=" * 60)
    print("检索器总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 语义搜索使用 VectorStoreRetriever
2. 关键词搜索使用 BM25Retriever
3. 提高召回率使用 MultiQueryRetriever
4. 综合效果使用 EnsembleRetriever
5. 根据场景自定义检索器

🔧 检索器参数:
- k: 返回结果数量
- fetch_k: MMR 预取数量
- lambda_mult: MMR 多样性权重
- score_threshold: 相似度阈值
    """)
    
    print("\n✅ 检索器学习完成！")


if __name__ == "__main__":
    main()
