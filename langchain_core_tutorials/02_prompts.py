#!/usr/bin/env python3
"""
LangChain Core 核心概念 02 - 提示词模板 (Prompts)
功能：掌握 ChatPromptTemplate 的使用

核心概念：
- ChatPromptTemplate: 聊天提示词模板
- MessagesPlaceholder: 消息占位符
- 变量替换: {variable} 语法
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def demo_basic_template():
    """演示基础提示词模板"""
    print("=" * 60)
    print("LangChain Core 核心概念 02 - 提示词模板")
    print("=" * 60)
    
    print("\n📋 基础提示词模板\n")
    print("-" * 50)
    
    # 方式 1: 使用 from_messages
    template1 = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}。"),
        ("user", "{question}")
    ])
    
    print("\n1️⃣ 使用 from_messages 创建模板:")
    print(f"   模板: {template1}")
    
    # 填充变量
    filled = template1.invoke({
        "role": "Python 专家",
        "question": "什么是列表推导式？"
    })
    
    print(f"\n   填充后:")
    for msg in filled.messages:
        print(f"   - {msg.type}: {msg.content}")
    
    # 方式 2: 使用消息对象
    template2 = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一位翻译专家。"),
        HumanMessagePromptTemplate.from_template("请将以下内容翻译成{language}: {text}")
    ])
    
    print("\n2️⃣ 使用消息对象创建模板:")
    filled2 = template2.invoke({
        "language": "英文",
        "text": "你好，世界"
    })
    
    print(f"   填充后:")
    for msg in filled2.messages:
        print(f"   - {msg.type}: {msg.content}")


def demo_messages_placeholder():
    """演示 MessagesPlaceholder"""
    print("\n" + "=" * 60)
    print("MessagesPlaceholder - 消息占位符")
    print("=" * 60)
    
    # 创建带占位符的模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一位有帮助的助手。记住对话历史。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    print("\n📍 MessagesPlaceholder 用途:\n")
    print("   在模板中预留一个位置，用于插入动态的消息列表")
    print("   常用于多轮对话中传递历史消息\n")
    
    # 模拟对话历史
    history = [
        HumanMessage(content="我叫张三"),
        AIMessage(content="你好张三！很高兴认识你。"),
        HumanMessage(content="我喜欢 Python 编程")
    ]
    
    filled = template.invoke({
        "history": history,
        "input": "我叫什么名字？"
    })
    
    print("📜 填充后的消息列表:\n")
    for msg in filled.messages:
        role = "🤖 系统" if msg.type == "system" else \
               "👤 用户" if msg.type == "human" else "🤖 AI"
        print(f"   {role}: {msg.content}")


def demo_few_shot_template():
    """演示少样本提示词模板"""
    print("\n" + "=" * 60)
    print("少样本学习 (Few-Shot Learning)")
    print("=" * 60)
    
    # 创建少样本模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析专家。判断文本的情感倾向。"),
        # 示例 1
        ("human", "这部电影太棒了！"),
        ("ai", "情感: 正面"),
        # 示例 2
        ("human", "服务态度太差了"),
        ("ai", "情感: 负面"),
        # 示例 3
        ("human", "今天天气一般"),
        ("ai", "情感: 中性"),
        # 实际输入
        ("human", "{text}")
    ])
    
    print("\n🎯 少样本提示词:\n")
    print("   通过提供示例，让模型学习任务模式\n")
    
    filled = template.invoke({"text": "这个产品超出我的预期！"})
    
    print("📜 完整提示词:\n")
    for msg in filled.messages:
        print(f"   {msg.type}: {msg.content}")
    
    # 使用模型
    print("\n🚀 调用模型:\n")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    response = llm.invoke(filled)
    print(f"   🤖 回复: {response.content}")


def demo_partial_template():
    """演示部分填充模板"""
    print("\n" + "=" * 60)
    print("部分填充模板 (Partial)")
    print("=" * 60)
    
    # 创建模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}，使用{language}回答。"),
        ("human", "{question}")
    ])
    
    print("\n📝 部分填充:\n")
    print("   先填充部分变量，剩余变量后续填充\n")
    
    # 部分填充
    partial_template = template.partial(role="历史专家", language="中文")
    
    print("   部分填充后: role='历史专家', language='中文'")
    print("   剩余变量: question\n")
    
    # 填充剩余变量
    filled = partial_template.invoke({"question": "谁是秦始皇？"})
    
    print("📜 最终填充结果:\n")
    for msg in filled.messages:
        print(f"   {msg.type}: {msg.content}")


def demo_prompt_composition():
    """演示提示词组合"""
    print("\n" + "=" * 60)
    print("提示词组合")
    print("=" * 60)
    
    # 创建基础模板
    base_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}。")
    ])
    
    # 创建问题模板
    question_template = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ])
    
    # 组合模板 (使用 + 操作符)
    combined = base_template + question_template
    
    print("\n➕ 使用 + 操作符合并模板:\n")
    
    filled = combined.invoke({
        "role": "科学专家",
        "question": "什么是量子力学？"
    })
    
    print("📜 组合后的消息:\n")
    for msg in filled.messages:
        print(f"   {msg.type}: {msg.content}")


def main():
    demo_basic_template()
    demo_messages_placeholder()
    demo_few_shot_template()
    demo_partial_template()
    demo_prompt_composition()
    
    print("\n" + "=" * 60)
    print("提示词模板总结")
    print("=" * 60)
    print("""
┌────────────────────────┬────────────────────────────────────┐
│ 类/方法                 │ 用途                               │
├────────────────────────┼────────────────────────────────────┤
│ ChatPromptTemplate     │ 创建聊天提示词模板                  │
│ from_messages()        │ 从消息列表创建模板                  │
│ MessagesPlaceholder    │ 消息占位符，用于动态消息列表         │
│ partial()              │ 部分填充变量                        │
│ + 操作符               │ 组合多个模板                        │
└────────────────────────┴────────────────────────────────────┘

💡 最佳实践:
1. 使用 from_messages() 创建结构化模板
2. 用 MessagesPlaceholder 处理对话历史
3. 少样本学习时提供 2-3 个示例
4. 部分填充提高模板复用性
    """)
    
    print("\n✅ 提示词模板学习完成！")


if __name__ == "__main__":
    main()
