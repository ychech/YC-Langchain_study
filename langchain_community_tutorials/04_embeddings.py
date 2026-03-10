#!/usr/bin/env python3
"""
LangChain Community 核心概念 04 - 嵌入模型 (Embeddings)
功能：学习使用各种文本嵌入模型

核心概念：
- OpenAIEmbeddings: OpenAI 的嵌入模型
- HuggingFaceEmbeddings: HuggingFace 的嵌入模型
- 嵌入用途: 相似度计算、聚类、分类
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def demo_huggingface_embeddings():
    """演示 HuggingFace 嵌入模型"""
    print("=" * 60)
    print("LangChain Community 核心概念 04 - 嵌入模型")
    print("=" * 60)
    
    print("\n🤗 HuggingFace Embeddings\n")
    print("-" * 50)
    
    print("\n1️⃣ 加载本地嵌入模型:")
    print("   模型: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("   特点: 多语言支持，轻量级\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 嵌入单个文本
    print("2️⃣ 嵌入单个文本:")
    text = "这是一个测试句子"
    vector = embeddings.embed_query(text)
    
    print(f"   文本: '{text}'")
    print(f"   向量维度: {len(vector)}")
    print(f"   向量前 5 个值: {vector[:5]}")
    
    # 批量嵌入
    print("\n3️⃣ 批量嵌入:")
    texts = [
        "机器学习是人工智能的分支",
        "深度学习使用神经网络",
        "自然语言处理让计算机理解语言"
    ]
    
    vectors = embeddings.embed_documents(texts)
    
    print(f"   文本数量: {len(texts)}")
    print(f"   向量形状: {len(vectors)} x {len(vectors[0])}")


def demo_similarity_calculation():
    """演示相似度计算"""
    print("\n" + "=" * 60)
    print("嵌入向量相似度计算")
    print("=" * 60)
    
    print("\n📐 使用余弦相似度计算文本相似性:\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 准备文本
    texts = [
        "机器学习",
        "深度学习",
        "人工智能",
        "Python 编程",
        "Java 编程"
    ]
    
    # 计算嵌入
    vectors = embeddings.embed_documents(texts)
    
    # 计算余弦相似度
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print("   文本相似度矩阵:\n")
    
    # 显示相似度
    print("   " + " ".join([f"{t:>12}" for t in texts]))
    for i, text in enumerate(texts):
        similarities = [cosine_similarity(vectors[i], vectors[j]) for j in range(len(texts))]
        print(f"   {text:>12} " + " ".join([f"{s:>12.3f}" for s in similarities]))
    
    print("\n   💡 观察: '机器学习' 和 '深度学习' 相似度最高")
    print("          'Python 编程' 和 'Java 编程' 相似度较高")


def demo_embedding_models_comparison():
    """嵌入模型对比"""
    print("\n" + "=" * 60)
    print("嵌入模型对比")
    print("=" * 60)
    
    print("""
┌─────────────────────────┬─────────────────┬─────────────────┐
│ 模型                     │ 维度            │ 特点            │
├─────────────────────────┼─────────────────┼─────────────────┤
│ text-embedding-3-small  │ 1536            │ OpenAI, 轻量    │
│ text-embedding-3-large  │ 3072            │ OpenAI, 高质量  │
│ paraphrase-multilingual │ 384             │ HuggingFace, 多语言│
│ BAAI/bge-large-zh       │ 1024            │ 中文优化        │
│ m3e-base                │ 768             │ 中文开源        │
└─────────────────────────┴─────────────────┴─────────────────┘

选择建议:
- 英文场景: OpenAI 或 all-MiniLM-L6-v2
- 中文场景: BAAI/bge-large-zh 或 m3e-base
- 多语言: paraphrase-multilingual-MiniLM-L12-v2
- 资源受限: 使用小维度模型 (384-768)
    """)


def demo_openai_embeddings():
    """演示 OpenAI 嵌入模型"""
    print("\n" + "=" * 60)
    print("OpenAI Embeddings")
    print("=" * 60)
    
    print("\n🔑 OpenAI 嵌入模型 (需要 API Key):\n")
    
    print("""
   from langchain_openai import OpenAIEmbeddings
   
   embeddings = OpenAIEmbeddings(
       model="text-embedding-3-small",  # 或 text-embedding-3-large
       api_key="your-api-key"
   )
   
   # 使用方式与 HuggingFace 相同
   vector = embeddings.embed_query("Hello world")
   vectors = embeddings.embed_documents(["text1", "text2"])
   
   特点:
   - text-embedding-3-small: 1536 维，性价比高
   - text-embedding-3-large: 3072 维，质量更好
    """)


def demo_embedding_cache():
    """演示嵌入缓存"""
    print("\n" + "=" * 60)
    print("嵌入缓存优化")
    print("=" * 60)
    
    print("\n💾 缓存嵌入结果避免重复计算:\n")
    
    print("""
   from langchain_core.embeddings import CacheBackedEmbeddings
   from langchain_core.storage import LocalFileStore
   
   # 创建底层嵌入模型
   underlying_embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   
   # 创建文件存储
   store = LocalFileStore("./cache/")
   
   # 创建缓存嵌入
   cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
       underlying_embeddings,
       store,
       namespace=underlying_embeddings.model_name
   )
   
   # 第一次计算会缓存
   vectors = cached_embeddings.embed_documents(["Hello world"])
   
   # 第二次直接从缓存读取
   vectors = cached_embeddings.embed_documents(["Hello world"])  # 更快!
    """)


def demo_embedding_applications():
    """演示嵌入的应用场景"""
    print("\n" + "=" * 60)
    print("嵌入的应用场景")
    print("=" * 60)
    
    applications = [
        {
            "场景": "语义搜索",
            "描述": "基于含义而非关键词匹配",
            "示例": "搜索'苹果'可以匹配'iPhone'、'水果'等"
        },
        {
            "场景": "文本聚类",
            "描述": "将相似文本分组",
            "示例": "新闻分类、客户反馈分组"
        },
        {
            "场景": "语义相似度",
            "描述": "计算文本间的相似程度",
            "示例": "重复内容检测、抄袭检测"
        },
        {
            "场景": "推荐系统",
            "描述": "基于内容相似度推荐",
            "示例": "相似文章推荐、产品推荐"
        },
        {
            "场景": "RAG 检索",
            "描述": "检索增强生成",
            "示例": "从知识库检索相关文档"
        }
    ]
    
    print()
    for i, app in enumerate(applications, 1):
        print(f"{i}. {app['场景']}")
        print(f"   描述: {app['描述']}")
        print(f"   示例: {app['示例']}\n")


def main():
    demo_huggingface_embeddings()
    demo_similarity_calculation()
    demo_embedding_models_comparison()
    demo_openai_embeddings()
    demo_embedding_cache()
    demo_embedding_applications()
    
    print("\n" + "=" * 60)
    print("嵌入模型总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 中文场景使用 BAAI/bge-large-zh 或 m3e-base
2. 多语言场景使用 paraphrase-multilingual
3. 资源受限选择小维度模型 (384-512)
4. 生产环境使用缓存避免重复计算
5. 根据任务选择合适的模型大小

⚠️ 注意事项:
- 嵌入模型需要下载，首次使用较慢
- 不同模型的向量维度不同
- 同一模型产生的向量才能比较
- 大模型质量高但计算慢
    """)
    
    print("\n✅ 嵌入模型学习完成！")


if __name__ == "__main__":
    main()
