# LangChain Community 核心概念学习

本目录包含 LangChain Community 的完整学习代码。LangChain Community 提供了丰富的第三方集成，包括文档加载器、向量存储、工具等。

## 📚 核心模块

| 模块 | 说明 |
|------|------|
| **Document Loaders** | 文档加载器 - PDF、Word、网页等 |
| **Vector Stores** | 向量存储 - FAISS、Chroma、Pinecone 等 |
| **Embeddings** | 嵌入模型 - OpenAI、HuggingFace 等 |
| **Tools** | 工具集成 - 搜索、计算、API 调用等 |
| **Retrievers** | 检索器 - 多种检索策略 |
| **Chat Models** | 聊天模型 - 各种 LLM 提供商 |

## 📁 文件结构

```
langchain_community_tutorials/
├── 01_document_loaders.py       # 文档加载器
├── 02_text_splitters.py         # 文本分割器
├── 03_vector_stores.py          # 向量存储
├── 04_embeddings.py             # 嵌入模型
├── 05_tools.py                  # 工具集成
├── 06_retrievers.py             # 检索器
├── 07_chat_models.py            # 聊天模型
├── 08_utilities.py              # 实用工具
├── sample_files/                # 示例文件
│   ├── sample.txt
│   └── sample.pdf
├── requirements.txt
└── README.md
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp ../.env .  # 或者创建新的 .env 文件

# 运行示例
python 01_document_loaders.py
```

## 📖 学习路径

1. **Document Loaders** - 学习加载各种格式的文档
2. **Text Splitters** - 掌握文本分割策略
3. **Vector Stores** - 了解向量数据库的使用
4. **Embeddings** - 学习文本嵌入
5. **Tools** - 掌握工具集成
6. **Retrievers** - 学习检索策略
7. **Chat Models** - 了解多种 LLM 集成
8. **Utilities** - 学习实用工具

## 🔗 参考资源

- [LangChain Community 文档](https://python.langchain.com/docs/integrations/)
