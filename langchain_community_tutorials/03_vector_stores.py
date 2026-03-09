#!/usr/bin/env python3
"""
LangChain Community 核心概念 03 - 向量存储 (Vector Stores)
功能：学习使用各种向量数据库

核心概念：
- FAISS: Facebook 的相似度搜索库
- Chroma: 开源向量数据库
- 向量检索: similarity_search, max_marginal_relevance_search
"""
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
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
        Document(page_content="Python 是一种高级编程语言，简洁优雅", metadata={"source": "doc1"}),
        Document(page_content="JavaScript 是网页开发的主要语言", metadata={"source": "doc2"}),
        Document(page_content="Java 是一种面向对象的编程语言", metadata={"source": "doc3"}),
        Document(page_content="Go 语言由 Google 开发，适合并发编程", metadata={"source": "doc4"}),
        Document(page_content="Rust 注重安全和性能，适合系统编程", metadata={"source": "doc5"}),
        Document(page_content="机器学习是人工智能的一个分支", metadata={"source": "doc6"}),
        Document(page_content="深度学习使用神经网络进行学习", metadata={"source": "doc7"}),
        Document(page_content="自然语言处理让计算机理解人类语言", metadata={"source": "doc8"}),
    ]


def get_embeddings():
    """获取嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )


def demo_faiss():
    """演示 FAISS 向量存储"""
    print("=" * 60)
    print("LangChain Community 核心概念 03 - 向量存储")
    print("=" * 60)
    
    print("\n🔍 FAISS - Facebook AI Similarity Search\n")
    print("-" * 50)
    
    # 准备数据
    docs = get_sample_documents()
    embeddings = get_embeddings()
    
    print("\n1️⃣ 创建 FAISS 向量库:")
    print("   加载嵌入模型...")
    
    # 从文档创建
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    print(f"   创建了 {len(docs)} 个文档的向量库")
    
    # 相似度搜索
    print("\n2️⃣ 相似度搜索:")
    query = "Python 编程"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"   查询: '{query}'")
    print(f"   返回前 3 个结果:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
        print(f"      来源: {doc.metadata['source']}\n")
    
    # 带分数的搜索
    print("3️⃣ 带相似度分数的搜索:")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    for doc, score in results_with_scores:
        print(f"   分数: {score:.4f} - {doc.page_content[:40]}...")
    
    # 保存和加载
    print("\n4️⃣ 保存和加载向量库:")
    save_path = "faiss_index"
    vectorstore.save_local(save_path)
    print(f"   已保存到: {save_path}")
    
    loaded_vectorstore = FAISS.load_local(
        save_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"   已从 {save_path} 加载")
    
    # 清理
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)


def demo_chroma():
    """演示 Chroma 向量存储"""
    print("\n" + "=" * 60)
    print("Chroma - 开源向量数据库")
    print("=" * 60)
    
    print("\n💾 Chroma 特点: 轻量、易用、支持持久化\n")
    
    docs = get_sample_documents()
    embeddings = get_embeddings()
    
    print("1️⃣ 创建 Chroma 向量库:")
    
    # 内存模式
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="my_collection"
    )
    
    print(f"   创建了 {len(docs)} 个文档的向量库")
    
    # 搜索
    print("\n2️⃣ 相似度搜索:")
    query = "人工智能"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"   查询: '{query}'\n")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content}")
    
    # MMR 搜索（多样性）
    print("\n3️⃣ MMR 搜索（最大边际相关性）:")
    print("   在相关性和多样性之间平衡\n")
    
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=5,  # 先获取 5 个，再选择 3 个
        lambda_mult=0.5  # 多样性权重
    )
    
    for i, doc in enumerate(mmr_results, 1):
        print(f"   {i}. {doc.page_content}")


def demo_retriever():
    """演示检索器"""
    print("\n" + "=" * 60)
    print("向量检索器 (Retriever)")
    print("=" * 60)
    
    print("\n🔎 将 VectorStore 转换为 Retriever:\n")
    
    docs = get_sample_documents()
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # 方式 1: 默认检索器
    print("1️⃣ 默认相似度检索器:")
    retriever = vectorstore.as_retriever()
    results = retriever.invoke("编程语言")
    
    print(f"   查询: '编程语言'")
    print(f"   返回 {len(results)} 个结果\n")
    
    # 方式 2: 配置检索参数
    print("2️⃣ 配置检索参数:")
    retriever = vectorstore.as_retriever(
        search_type="similarity",      # 搜索类型
        search_kwargs={"k": 2}         # 返回数量
    )
    results = retriever.invoke("机器学习")
    
    for doc in results:
        print(f"   - {doc.page_content}")
    
    # 方式 3: MMR 检索器
    print("\n3️⃣ MMR 检索器:")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5}
    )
    results = retriever.invoke("AI 技术")
    
    for doc in results:
        print(f"   - {doc.page_content}")


def demo_vector_store_operations():
    """演示向量库操作"""
    print("\n" + "=" * 60)
    print("向量库操作")
    print("=" * 60)
    
    print("\n🛠️ 常用操作:\n")
    
    docs = get_sample_documents()
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs[:4], embeddings)
    
    print("1️⃣ 添加文档:")
    new_docs = docs[4:6]
    vectorstore.add_documents(new_docs)
    print(f"   添加了 {len(new_docs)} 个文档")
    print(f"   当前总数: {len(docs[:4]) + len(new_docs)}")
    
    print("\n2️⃣ 添加文本:")
    texts = ["C++ 是一种高性能编程语言"]
    metadatas = [{"source": "doc9"}]
    vectorstore.add_texts(texts, metadatas)
    print(f"   添加了 {len(texts)} 个文本")
    
    print("\n3️⃣ 删除文档:")
    print("   注意: FAISS 不支持直接删除，需要重建索引")
    print("   Chroma 支持通过 ID 删除")


def show_vector_store_comparison():
    """向量存储对比"""
    print("\n" + "=" * 60)
    print("向量存储对比")
    print("=" * 60)
    
    print("""
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ 特性             │ FAISS           │ Chroma          │ Pinecone        │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 类型             │ 本地库          │ 本地/服务       │ 云服务          │
│ 持久化           │ 支持            │ 支持            │ 自动            │
│ 性能             │ 高              │ 中              │ 高              │
│ 易用性           │ 中              │ 高              │ 高              │
│ 功能丰富度       │ 中              │ 高              │ 高              │
│ 适用场景         │ 本地开发        │ 中小项目        │ 生产环境        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

其他向量数据库:
- Weaviate: 开源，GraphQL 接口
- Qdrant: Rust 编写，高性能
- Milvus: 分布式，大规模数据
- pgvector: PostgreSQL 扩展
    """)


def main():
    demo_faiss()
    demo_chroma()
    demo_retriever()
    demo_vector_store_operations()
    show_vector_store_comparison()
    
    print("\n" + "=" * 60)
    print("向量存储总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 本地开发使用 FAISS 或 Chroma
2. 生产环境考虑 Pinecone 或 Weaviate
3. 根据数据量选择合适的向量库
4. 合理设置 chunk_size 和 overlap
5. 使用 MMR 检索提高结果多样性

🎯 使用流程:
1. 准备文档
2. 选择嵌入模型
3. 创建 VectorStore
4. 转换为 Retriever
5. 进行相似度搜索
    """)
    
    print("\n✅ 向量存储学习完成！")


if __name__ == "__main__":
    main()
