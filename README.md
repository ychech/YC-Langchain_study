# 项目代码整理

本仓库包含多个 AI/ML 相关的学习和项目代码，已按功能分类整理。

## 📁 目录结构

```
Lc/
├── langchain_core_tutorials/   # LangChain Core 核心组件教程
│   ├── 01_messages.py
│   ├── 02_prompts.py
│   ├── 03_models.py
│   ├── 04_output_parsers.py
│   ├── 05_runnable_interface.py
│   ├── 06_lcel_basics.py
│   ├── 07_lcel_advanced.py
│   ├── 08_callbacks.py
│   ├── requirements.txt
│   └── README.md
│
├── langchain_community_tutorials/  # LangChain Community 集成教程
│   ├── 01_document_loaders.py
│   ├── 02_text_splitters.py
│   ├── 03_vector_stores.py
│   ├── 04_embeddings.py
│   ├── 05_tools.py
│   ├── 06_retrievers.py
│   ├── 07_chat_models.py
│   ├── 08_utilities.py
│   ├── requirements.txt
│   └── README.md
│
├── langchain_templates/        # LangChain 1.1.0 快速上手代码模板
│   ├── 01_basic_llm.py         # 基础 LLM 调用
│   ├── 02_memory_chat.py       # 带记忆的对话
│   ├── 03_rag_basic.py         # 基础 RAG 实现
│   ├── 04_deep_agent.py        # Deep Agent 智能代理
│   ├── 05_langsmith_tracing.py # LangSmith 追踪
│   ├── 06_complete_template.py # 完整综合模板
│   ├── requirements.txt
│   └── README.md
│
├── langgraph_tutorials/        # LangGraph 工作流教程
│   ├── 01_state_graph_basic.py
│   ├── 02_conditional_edges.py
│   ├── 03_state_management.py
│   ├── 04_memory_persistence.py
│   ├── 05_human_in_the_loop.py
│   ├── 06_multi_agent.py
│   ├── 07_rag_with_langgraph.py
│   ├── 08_streaming_events.py
│   ├── requirements.txt
│   └── README.md
│
├── daily_news.html             # 生成的日报示例
├── .env                        # 环境变量配置
└── README.md                   # 本文件
```

## 🚀 快速开始

### LangChain 模板

```bash
cd langchain_templates
pip install -r requirements.txt

# 运行示例
python 01_basic_llm.py
python 04_deep_agent.py
```

### LangChain Core 教程

```bash
cd langchain_core_tutorials
pip install -r requirements.txt
python 01_messages.py
```

### LangChain Community 教程

```bash
cd langchain_community_tutorials
pip install -r requirements.txt
python 01_document_loaders.py
```

### LangGraph 教程

```bash
cd langgraph_tutorials
pip install -r requirements.txt
python 01_state_graph_basic.py
```

## 📚 主要内容

### 1. LangChain Core 核心组件

- ✅ Messages - 消息类型与处理
- ✅ Prompts - 提示词模板
- ✅ Models - 模型调用
- ✅ Output Parsers - 输出解析
- ✅ Runnable Interface - 可运行接口
- ✅ LCEL - LangChain 表达式语言
- ✅ Callbacks - 回调机制

### 2. LangChain Community 集成

- ✅ Document Loaders - 文档加载器
- ✅ Text Splitters - 文本分割器
- ✅ Vector Stores - 向量存储
- ✅ Embeddings - 嵌入模型
- ✅ Tools - 工具集成
- ✅ Retrievers - 检索器
- ✅ Chat Models - 聊天模型
- ✅ Utilities - 实用工具

### 3. LangChain 模板

快速上手的完整代码模板：
- 基础 LLM 调用
- 带记忆的对话
- RAG 检索增强生成
- Deep Agent 智能代理
- LangSmith 追踪
- 完整综合模板

### 4. LangGraph 工作流

- State Graph 基础
- 条件边
- 状态管理
- 记忆持久化
- 人工介入
- 多智能体系统
- RAG 与 LangGraph 结合
- 流式事件

## ⚙️ 环境配置

创建 `.env` 文件：

```env
# DeepSeek API
DEEPSEEK_API_KEY=your_key

# Kimi API
KIMI_API_KEY=your_key

# LangSmith (可选)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=your_project
```

## 📖 学习路径

### LangChain 入门
1. **核心基础**：`langchain_core_tutorials/01_messages.py`
2. **提示词**：`langchain_core_tutorials/02_prompts.py`
3. **模型调用**：`langchain_core_tutorials/03_models.py`
4. **输出解析**：`langchain_core_tutorials/04_output_parsers.py`
5. **LCEL 基础**：`langchain_core_tutorials/06_lcel_basics.py`

### LangChain 进阶
1. **文档加载**：`langchain_community_tutorials/01_document_loaders.py`
2. **向量存储**：`langchain_community_tutorials/03_vector_stores.py`
3. **RAG 检索**：`langchain_community_tutorials/06_retrievers.py`

### 实战模板
1. **入门**：`langchain_templates/01_basic_llm.py`
2. **进阶**：`langchain_templates/02_memory_chat.py`
3. **RAG**：`langchain_templates/03_rag_basic.py`
4. **Agent**：`langchain_templates/04_deep_agent.py`

### LangGraph 工作流
1. **基础**：`langgraph_tutorials/01_state_graph_basic.py`
2. **状态管理**：`langgraph_tutorials/03_state_management.py`
3. **多智能体**：`langgraph_tutorials/06_multi_agent.py`

## 📝 说明

- `langchain_templates/` 使用 LangChain 1.1.0 最新 API
- `langchain_core_tutorials/` 专注于核心组件学习
- `langchain_community_tutorials/` 涵盖第三方集成
- `langgraph_tutorials/` 工作流编排与多智能体
- 所有代码均包含详细注释
