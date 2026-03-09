# LangChain Core 核心概念学习

本目录包含 LangChain Core 核心概念的完整学习代码。LangChain Core 是整个 LangChain 生态的基础，提供了与模型交互的核心抽象。

## 📚 核心概念

LangChain Core 提供了与 LLM 交互的基础组件，不依赖任何特定框架。

| 概念 | 说明 |
|------|------|
| **Messages** | 消息类型 - SystemMessage, HumanMessage, AIMessage |
| **Prompts** | 提示词模板 - ChatPromptTemplate |
| **Models** | 模型接口 - BaseChatModel |
| **Output Parsers** | 输出解析器 - 结构化输出 |
| **Runnable** | 可运行对象 - 核心抽象接口 |
| **LCEL** | LangChain Expression Language - 链式调用语法 |

## 📁 文件结构

```
langchain_core_tutorials/
├── 01_messages.py               # 消息类型详解
├── 02_prompts.py                # 提示词模板
├── 03_models.py                 # 模型接口与调用
├── 04_output_parsers.py         # 输出解析器
├── 05_runnable_interface.py     # Runnable 核心接口
├── 06_lcel_basics.py            # LCEL 基础
├── 07_lcel_advanced.py          # LCEL 进阶
├── 08_callbacks.py              # 回调机制
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
python 01_messages.py
```

## 📖 学习路径

1. **Messages** - 理解消息类型和角色
2. **Prompts** - 掌握提示词模板
3. **Models** - 学习模型接口
4. **Output Parsers** - 结构化输出
5. **Runnable Interface** - 核心抽象
6. **LCEL Basics** - 链式调用基础
7. **LCEL Advanced** - 进阶用法
8. **Callbacks** - 回调和事件监听

## 🔗 参考资源

- [LangChain Core 文档](https://api.python.langchain.com/en/stable/core_api_reference.html)
- [LangChain 官方文档](https://python.langchain.com/)
