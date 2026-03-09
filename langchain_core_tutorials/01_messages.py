#!/usr/bin/env python3
"""
LangChain Core 核心概念 01 - 消息类型 (Messages)
功能：理解不同类型的消息及其作用

核心概念：
- SystemMessage: 系统消息，设定 AI 角色和行为
- HumanMessage: 人类消息，用户输入
- AIMessage: AI 消息，模型回复
- ToolMessage: 工具消息，工具调用结果
"""
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_to_dict,
    messages_from_dict
)
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


def demo_message_types():
    """演示不同消息类型"""
    print("=" * 60)
    print("LangChain Core 核心概念 01 - 消息类型")
    print("=" * 60)
    
    # ========== 1. 创建不同类型的消息 ==========
    print("\n📨 创建不同类型的消息\n")
    print("-" * 50)
    
    # SystemMessage: 设定 AI 的系统提示词
    system_msg = SystemMessage(content="你是一位专业的 Python 编程助手，回答简洁明了。")
    print(f"\n1️⃣ SystemMessage (系统消息):")
    print(f"   类型: {type(system_msg).__name__}")
    print(f"   内容: {system_msg.content}")
    print(f"   作用: 设定 AI 的角色和行为")
    
    # HumanMessage: 用户输入
    human_msg = HumanMessage(content="什么是装饰器？")
    print(f"\n2️⃣ HumanMessage (人类消息):")
    print(f"   类型: {type(human_msg).__name__}")
    print(f"   内容: {human_msg.content}")
    print(f"   作用: 用户的输入/问题")
    
    # AIMessage: AI 回复
    ai_msg = AIMessage(content="装饰器是 Python 中用于修改函数或类行为的语法糖。")
    print(f"\n3️⃣ AIMessage (AI 消息):")
    print(f"   类型: {type(ai_msg).__name__}")
    print(f"   内容: {ai_msg.content}")
    print(f"   作用: AI 的回复/输出")
    
    # ToolMessage: 工具调用结果
    tool_msg = ToolMessage(
        content="计算结果: 42",
        tool_call_id="call_123",
        name="calculator"
    )
    print(f"\n4️⃣ ToolMessage (工具消息):")
    print(f"   类型: {type(tool_msg).__name__}")
    print(f"   内容: {tool_msg.content}")
    print(f"   工具ID: {tool_msg.tool_call_id}")
    print(f"   工具名: {tool_msg.name}")
    print(f"   作用: 工具调用的返回结果")


def demo_message_usage():
    """演示消息的实际使用"""
    print("\n" + "=" * 60)
    print("消息的实际使用")
    print("=" * 60)
    
    # 构建消息列表
    messages = [
        SystemMessage(content="你是一位友好的助手。"),
        HumanMessage(content="你好！"),
        AIMessage(content="你好！很高兴见到你。有什么可以帮助你的吗？"),
        HumanMessage(content="今天天气怎么样？")
    ]
    
    print("\n📜 消息对话历史:\n")
    for i, msg in enumerate(messages, 1):
        role = "🤖 系统" if isinstance(msg, SystemMessage) else \
               "👤 用户" if isinstance(msg, HumanMessage) else \
               "🤖 AI" if isinstance(msg, AIMessage) else "🔧 工具"
        print(f"{i}. {role}: {msg.content[:50]}...")
    
    # 使用模型
    print("\n🚀 使用消息列表调用模型:\n")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    response = llm.invoke(messages)
    print(f"🤖 AI 回复: {response.content}")


def demo_message_serialization():
    """演示消息的序列化和反序列化"""
    print("\n" + "=" * 60)
    print("消息序列化 (用于保存和传输)")
    print("=" * 60)
    
    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好"),
        AIMessage(content="你好！")
    ]
    
    # 转换为字典（可 JSON 序列化）
    print("\n📦 转换为字典:\n")
    dict_list = messages_to_dict(messages)
    for msg_dict in dict_list:
        print(f"   {msg_dict}")
    
    # 从字典恢复
    print("\n📥 从字典恢复:\n")
    restored_messages = messages_from_dict(dict_list)
    for msg in restored_messages:
        print(f"   {type(msg).__name__}: {msg.content}")


def demo_message_properties():
    """演示消息的属性和方法"""
    print("\n" + "=" * 60)
    print("消息属性和方法")
    print("=" * 60)
    
    msg = AIMessage(
        content="这是一个 AI 回复",
        additional_kwargs={"confidence": 0.95},
        response_metadata={"model": "gpt-4"}
    )
    
    print(f"\n📋 消息属性:\n")
    print(f"   content: {msg.content}")
    print(f"   type: {msg.type}")
    print(f"   additional_kwargs: {msg.additional_kwargs}")
    print(f"   response_metadata: {msg.response_metadata}")
    print(f"   id: {msg.id}")


def main():
    demo_message_types()
    demo_message_usage()
    demo_message_serialization()
    demo_message_properties()
    
    print("\n" + "=" * 60)
    print("消息类型总结")
    print("=" * 60)
    print("""
┌─────────────────┬─────────────────────────────────────────┐
│ 消息类型         │ 用途                                    │
├─────────────────┼─────────────────────────────────────────┤
│ SystemMessage   │ 设定 AI 角色、行为、约束                 │
│ HumanMessage    │ 用户输入/问题                           │
│ AIMessage       │ AI 回复/输出                            │
│ ToolMessage     │ 工具调用结果                            │
└─────────────────┴─────────────────────────────────────────┘

💡 最佳实践:
1. SystemMessage 放在消息列表最前面
2. HumanMessage 和 AIMessage 交替出现
3. ToolMessage 紧跟在 AIMessage 之后
    """)
    
    print("\n✅ 消息类型学习完成！")


if __name__ == "__main__":
    main()
