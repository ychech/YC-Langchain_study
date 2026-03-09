# LangChain 1.0.x 快速上手代码模板

本目录包含 LangChain 1.0.x 的完整快速上手代码模板，涵盖 Deep Agent、RAG、LangSmith 追踪等核心功能。

## ⚠️ 重要说明

LangChain 1.0 于 **2025年10月** 发布，是首个稳定版本，与 0.x 版本有较大的 API 差异：

| 特性 | 0.3.x | 1.0.x |
|------|-------|-------|
| Agent 创建 | `create_openai_tools_agent` + `AgentExecutor` | `create_agent`（简化） |
| 返回值 | `{"output": ...}` | `{"messages": [...]}` |
| RAG 组件 | `langchain` | `langchain-classic` |
| 记忆管理 | `RunnableWithMessageHistory` | `trim_messages` + 手动管理 |

## 📁 文件结构

```
langchain_templates/
├── 01_basic_llm.py          # 基础 LLM 调用（LCEL 链式调用）
├── 02_memory_chat.py        # 带记忆的对话（trim_messages）
├── 03_rag_basic.py          # 基础 RAG 实现（langchain-classic）
├── 04_deep_agent.py         # Deep Agent（新 create_agent API）
├── 05_langsmith_tracing.py  # LangSmith 追踪
├── 06_complete_template.py  # 完整综合模板
├── requirements.txt         # 依赖（1.0.x 版本）
└── README.md                # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖（LangChain 1.0.x）
pip install -r requirements.txt
```

requirements.txt:
```
langchain>=1.0.0
langchain-core>=0.3.0
langchain-openai>=0.3.0
langchain-classic>=0.1.0
langchain-text-splitters>=0.3.0
langchain-huggingface>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key

# LangSmith 追踪（可选）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```

### 3. 运行示例

```bash
# 基础 LLM 调用
python 01_basic_llm.py

# 带记忆的对话
python 02_memory_chat.py

# RAG 检索增强生成
python 03_rag_basic.py

# Deep Agent 智能代理
python 04_deep_agent.py

# LangSmith 追踪
python 05_langsmith_tracing.py

# 完整综合模板
python 06_complete_template.py
```

## 📖 功能详解

### 01_basic_llm.py - 基础 LLM 调用
- 直接调用 DeepSeek 模型
- LCEL（LangChain Expression Language）链式调用
- 批量调用 `batch()`

### 02_memory_chat.py - 带记忆的对话
- 使用 `trim_messages` 控制上下文长度
- 手动管理对话历史
- 避免 token 超限

### 03_rag_basic.py - 基础 RAG 实现
- **注意**：RAG 相关功能在 `langchain-classic` 包中
- 文档分割：`langchain-text-splitters`
- 向量库：FAISS
- 嵌入模型：HuggingFace

### 04_deep_agent.py - Deep Agent 智能代理
- **新 API**：`from langchain.agents import create_agent`
- 使用 `@tool` 装饰器定义工具
- 返回值格式：`{"messages": [...]}`

```python
# LangChain 1.0 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="..."
)

# 调用并获取结果
response = agent.invoke({"input": question})
messages = response["messages"]
ai_response = messages[-1].content
```

### 05_langsmith_tracing.py - LangSmith 追踪
- 运行监控和调试
- 使用 `config` 添加元数据
- 批量调用追踪

### 06_complete_template.py - 完整综合模板
- 模块化设计
- 整合所有功能
- 生产级代码结构

## 🔧 核心概念

### LCEL（LangChain Expression Language）

```python
from langchain_core.runnables import RunnablePassthrough

# 使用 | 操作符构建链
chain = prompt | llm | output_parser

# 带条件的链
chain = (
    {"input": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
```

### Agent 创建（1.0 新方式）

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """工具描述"""
    return "结果"

agent = create_agent(
    model=llm,
    tools=[my_tool],
    system_prompt="系统提示词"
)

# 调用
response = agent.invoke({"input": "用户输入"})
ai_message = response["messages"][-1]
```

### 记忆管理（1.0 新方式）

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=2000,
    strategy="last",
    token_counter=llm
)

# 在链中使用
trimmed = trimmer.invoke(messages)
```

## 📚 参考资源

- [LangChain 1.0 官方文档](https://docs.langchain.com/)
- [LangChain 1.0 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [LangSmith 平台](https://smith.langchain.com)
- [DeepSeek API 文档](https://platform.deepseek.com/)

## ⚠️ 注意事项

1. **API Key 安全**：不要将 API Key 硬编码在代码中
2. **版本兼容**：LangChain 1.0 与 0.x 不兼容，需要重新学习
3. **RAG 组件**：需要使用 `langchain-classic` 包
4. **Agent 返回值**：从 `{"output": ...}` 变为 `{"messages": [...]}`
5. **Python 版本**：LangChain 1.0 需要 Python 3.10+

## 📝 更新日志

- 2025-03-06: 更新到 LangChain 1.0.x 版本
