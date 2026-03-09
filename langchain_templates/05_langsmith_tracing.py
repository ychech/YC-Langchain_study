#!/usr/bin/env python3
"""
LangChain 1.0.x 快速上手 - LangSmith 追踪
功能：监控、调试和优化 LangChain 应用
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量（支持从父目录加载 .env）
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # 尝试从当前目录加载

# ========== 配置 LangSmith ==========
# 在 .env 文件中设置以下变量：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key
# LANGCHAIN_PROJECT=your_project_name

# 验证 LangSmith 配置
tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if tracing_enabled:
    print("✅ LangSmith 追踪已启用")
    print(f"   项目：{os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("⚠️ LangSmith 追踪未启用")
    print("   请在 .env 文件中设置：")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY=your_key")
    print("   LANGCHAIN_PROJECT=your_project_name")
print()

# ========== 初始化模型 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta/v1",
    temperature=0.7,
    max_tokens=2048
)

# ========== 示例1：基础链追踪 ==========
print("===== 示例1：基础链追踪 =====\n")

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将中文翻译成英文。"),
    ("user", "请翻译：{text}")
])

chain1 = prompt1 | llm | StrOutputParser()

result1 = chain1.invoke({"text": "你好，世界！"})
print(f"翻译结果：{result1}\n")

# ========== 示例2：复杂链追踪 ==========
print("===== 示例2：复杂链追踪 =====\n")

# 构建一个多步骤的链
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个内容生成助手。"),
    ("user", "请为'{topic}'写一段100字的简介。")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("system", "你是一个标题生成专家。"),
    ("user", "请为以下内容生成3个吸引人的标题：\n{content}")
])

# 组合链：先生成内容，再生成标题
content_chain = prompt2 | llm | StrOutputParser()

def generate_titles(data):
    """生成标题"""
    return (prompt3 | llm | StrOutputParser()).invoke({"content": data["content"]})

# 使用 RunnablePassthrough 传递数据
full_chain = (
    {"topic": RunnablePassthrough()}
    | RunnablePassthrough.assign(content=content_chain)
    | RunnablePassthrough.assign(titles=generate_titles)
)

result2 = full_chain.invoke("人工智能")
print(f"主题：人工智能")
print(f"内容：{result2['content'][:100]}...")
print(f"标题：{result2['titles']}\n")

# ========== 示例3：批量调用追踪 ==========
print("===== 示例3：批量调用追踪 =====\n")

topics = ["机器学习", "深度学习", "自然语言处理"]
results = chain1.batch([{"text": t} for t in topics])

for topic, result in zip(topics, results):
    print(f"{topic} -> {result}")

# ========== 示例4：带元数据的追踪 ==========
print("\n===== 示例4：带元数据的追踪 =====\n")

# 使用 config 添加额外的追踪元数据
result4 = chain1.invoke(
    {"text": "LangSmith 是 LangChain 的可观测性平台"},
    config={
        "run_name": "translation_run",  # 自定义运行名称
        "tags": ["translation", "chinese-to-english"],  # 标签
        "metadata": {  # 自定义元数据
            "user_id": "user_123",
            "version": "1.0",
            "environment": "development"
        }
    }
)
print(f"结果：{result4}\n")

# ========== 查看追踪结果 ==========
print("="*60)
print("追踪完成！")
if tracing_enabled:
    print(f"请访问 https://smith.langchain.com 查看详细追踪信息")
    print(f"项目：{os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("LangSmith 未启用，无法查看追踪信息")
