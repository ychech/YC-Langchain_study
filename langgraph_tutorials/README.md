# LangGraph 1.0.x 核心概念学习

本目录包含 LangGraph 核心概念的完整学习代码，从基础到进阶。

## 📚 核心概念

LangGraph 是 LangChain 的扩展，用于构建**有状态、多步骤、循环**的 AI 应用。

| 概念 | 说明 |
|------|------|
| **State** | 状态 - 整个图的共享数据 |
| **Node** | 节点 - 执行具体任务的函数 |
| **Edge** | 边 - 连接节点，控制流程 |
| **Graph** | 图 - 整个工作流的定义 |
| **Checkpoint** | 检查点 - 保存状态，支持持久化 |

## 📁 文件结构

```
langgraph_tutorials/
├── 01_state_graph_basic.py      # StateGraph 基础
├── 02_conditional_edges.py       # 条件边（分支逻辑）
├── 03_state_management.py        # 状态管理
├── 04_memory_persistence.py      # 记忆与持久化
├── 05_human_in_the_loop.py       # 人机协同
├── 06_multi_agent.py             # 多 Agent 系统
├── 07_rag_with_langgraph.py      # RAG 完整实现
├── 08_streaming_events.py        # 流式事件处理
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
python 01_state_graph_basic.py
```

## 📖 学习路径

1. **StateGraph 基础** - 理解图的基本结构
2. **条件边** - 实现 if/else 分支逻辑
3. **状态管理** - 掌握状态传递和更新
4. **记忆与持久化** - 保存对话历史
5. **人机协同** - 人工审核和干预
6. **多 Agent 系统** - 多个 Agent 协作
7. **RAG 完整实现** - 结合向量检索
8. **流式事件** - 实时响应处理

## 🔗 参考资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 概念指南](https://langchain-ai.github.io/langgraph/concepts/)
