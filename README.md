# 项目代码整理

本仓库包含多个 AI/ML 相关的学习和项目代码，已按功能分类整理。

## 📁 目录结构

```
Lc/
├── langchain_templates/     # LangChain 1.1.0 快速上手代码模板
│   ├── 01_basic_llm.py          # 基础 LLM 调用
│   ├── 02_memory_chat.py        # 带记忆的对话
│   ├── 03_rag_basic.py          # 基础 RAG 实现
│   ├── 04_deep_agent.py         # Deep Agent 智能代理
│   ├── 05_langsmith_tracing.py  # LangSmith 追踪
│   ├── 06_complete_template.py  # 完整综合模板
│   ├── requirements.txt
│   └── README.md
│
├── attention_demo/          # 注意力机制学习代码
│   ├── 01_self_attention_numpy.py
│   ├── 02_self_attention_with_positional_encoding.py
│   ├── 03_transformer_encoder_pytorch.py
│   ├── 04_multi_head_attention.py
│   └── README.md
│
├── news_generator/          # AI 新闻日报生成器
│   ├── daily_news_generator.py
│   └── README.md
│
├── retrieval-augmented-generation/  # RAG 相关代码
│
├── archive/                 # 归档的旧代码
│   ├── 01_basic_llm_old.py
│   ├── 02_memory_chat_old.py
│   ├── 03_agent_with_tools_old.py
│   └── README.md
│
├── daily_news.html          # 生成的日报示例
├── .env                     # 环境变量配置
└── README.md                # 本文件
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

### 注意力机制学习

```bash
cd attention_demo
python 01_self_attention_numpy.py
```

### 新闻生成器

```bash
cd news_generator
python daily_news_generator.py
```

## 📚 主要内容

### 1. LangChain 1.1.0 模板

包含完整的快速上手代码：
- ✅ 基础 LLM 调用
- ✅ 带记忆的对话
- ✅ RAG 检索增强生成
- ✅ Deep Agent 智能代理
- ✅ LangSmith 追踪
- ✅ 完整综合模板

### 2. 注意力机制

从 NumPy 基础实现到 PyTorch 完整 Transformer：
- 自注意力机制
- 位置编码
- 多头注意力
- Transformer Encoder

### 3. 新闻生成器

基于 Kimi API 的智能新闻日报生成工具。

## ⚙️ 环境配置

创建 `.env` 文件：

```env
# DeepSeek API
DEEPSEEK_API_KEY=your_key

# Kimi API (新闻生成器)
KIMI_API_KEY=your_key

# LangSmith (可选)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=your_project
```

## 📖 学习路径

1. **入门**：`langchain_templates/01_basic_llm.py`
2. **进阶**：`langchain_templates/02_memory_chat.py`
3. **RAG**：`langchain_templates/03_rag_basic.py`
4. **Agent**：`langchain_templates/04_deep_agent.py`
5. **监控**：`langchain_templates/05_langsmith_tracing.py`
6. **综合**：`langchain_templates/06_complete_template.py`

## 📝 说明

- `archive/` 目录存放旧版本代码，仅供参考
- `langchain_templates/` 使用 LangChain 1.1.0 最新 API
- 所有代码均包含详细注释
