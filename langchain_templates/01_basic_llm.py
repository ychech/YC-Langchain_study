#!/usr/bin/env python3
"""
LangChain 1.0.x 快速上手 - 基础 LLM 调用
功能：最基本的 DeepSeek 模型调用示例
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# 加载环境变量,用于从 .env 文件中读取敏感配置信息
load_dotenv()

# 初始化 DeepSeek 大模型
llm = ChatOpenAI(
    model="deepseek-chat",  # 写代码用 deepseek-coder 效果更好
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta/v1",
    temperature=0.7,
    max_tokens=2048
)

# 方式1：直接调用
print("===== 方式1：直接调用 =====")
response = llm.invoke("你好，请介绍一下自己")
print(f"回答：{response.content}\n")

# 方式2：结合提示词模板（LCEL 链式调用）
print("===== 方式2：提示词模板 + 链式调用 =====")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的Python编程助手，回答简洁明了。"),
    ("user", "{question}")
])

# 使用 | 操作符构建链（LCEL - LangChain Expression Language）
chain = prompt | llm

# 调用链
chain_response = chain.invoke({"question": "Python 中列表和元组有什么区别？"})
print(f"回答：{chain_response.content}\n")

# 方式3：批量调用
print("===== 方式3：批量调用 =====")
questions = [
    {"question": "什么是装饰器？"},
    {"question": "解释一下生成器"},
    {"question": "Python 的 GIL 是什么？"}
]
batch_responses = chain.batch(questions)
for q, r in zip(questions, batch_responses):
    print(f"Q: {q['question']}")
    print(f"A: {r.content[:50]}...\n")

print("✅ 基础 LLM 调用示例完成")
