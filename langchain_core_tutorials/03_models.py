#!/usr/bin/env python3
"""
LangChain Core 核心概念 03 - 模型接口 (Models)
功能：掌握 BaseChatModel 的使用和配置

核心概念：
- BaseChatModel: 聊天模型基类
- invoke: 同步调用
- ainvoke: 异步调用
- batch: 批量调用
- stream: 流式调用
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

# 加载环境变量
load_dotenv()


def demo_model_creation():
    """演示模型创建和配置"""
    print("=" * 60)
    print("LangChain Core 核心概念 03 - 模型接口")
    print("=" * 60)
    
    print("\n🔧 模型创建和配置\n")
    print("-" * 50)
    
    # 方式 1: 基础配置
    llm1 = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    print("\n1️⃣ 基础配置:")
    print(f"   模型: {llm1.model_name}")
    print(f"   温度: {llm1.temperature}")
    
    # 方式 2: 完整配置
    llm2 = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,      # 创造性程度 (0-2)
        max_tokens=2048,      # 最大生成 token 数
        timeout=30,           # 超时时间
        max_retries=3,        # 最大重试次数
    )
    print("\n2️⃣ 完整配置:")
    print(f"   模型: {llm2.model_name}")
    print(f"   温度: {llm2.temperature}")
    print(f"   最大 tokens: {llm2.max_tokens}")


def demo_invoke():
    """演示同步调用 invoke"""
    print("\n" + "=" * 60)
    print("同步调用 - invoke()")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n📤 调用方式 1: 字符串输入\n")
    response1 = llm.invoke("你好！")
    print(f"   输入: '你好！'")
    print(f"   输出: {response1.content}")
    
    print("\n📤 调用方式 2: 消息列表输入\n")
    messages = [
        SystemMessage(content="你是一位友好的助手。"),
        HumanMessage(content="什么是 Python？")
    ]
    response2 = llm.invoke(messages)
    print(f"   输入: [SystemMessage, HumanMessage]")
    print(f"   输出: {response2.content[:80]}...")


def demo_batch():
    """演示批量调用 batch"""
    print("\n" + "=" * 60)
    print("批量调用 - batch()")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n📦 批量处理多个输入:\n")
    
    inputs = [
        "什么是机器学习？",
        "什么是深度学习？",
        "什么是神经网络？"
    ]
    
    print(f"   输入数量: {len(inputs)}")
    print("   批量调用中...")
    
    responses = llm.batch(inputs)
    
    print("\n📥 批量结果:\n")
    for i, (inp, resp) in enumerate(zip(inputs, responses), 1):
        print(f"   {i}. 输入: {inp}")
        print(f"      输出: {resp.content[:50]}...\n")


def demo_stream():
    """演示流式调用 stream"""
    print("\n" + "=" * 60)
    print("流式调用 - stream()")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n🌊 流式输出 (逐字显示):\n")
    
    prompt = "用一句话介绍 Python"
    print(f"   输入: {prompt}")
    print(f"   输出: ", end="", flush=True)
    
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
    
    print()  # 换行


async def demo_async():
    """演示异步调用 ainvoke"""
    print("\n" + "=" * 60)
    print("异步调用 - ainvoke()")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n⚡ 异步调用:\n")
    
    # 单个异步调用
    print("   单个异步调用...")
    response = await llm.ainvoke("什么是异步编程？")
    print(f"   结果: {response.content[:50]}...")
    
    # 并发多个异步调用
    print("\n   并发多个异步调用...")
    tasks = [
        llm.ainvoke("什么是 AI？"),
        llm.ainvoke("什么是 ML？"),
        llm.ainvoke("什么是 DL？")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, resp in enumerate(responses, 1):
        print(f"   {i}. {resp.content[:40]}...")


def demo_model_parameters():
    """演示模型参数的影响"""
    print("\n" + "=" * 60)
    print("模型参数对比")
    print("=" * 60)
    
    prompt = "写一句关于秋天的诗"
    
    print(f"\n📝 提示词: '{prompt}'\n")
    
    # 不同温度参数
    temperatures = [0.0, 0.7, 1.5]
    
    for temp in temperatures:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temp
        )
        
        response = llm.invoke(prompt)
        print(f"   温度 {temp}: {response.content}")


def demo_response_metadata():
    """演示响应元数据"""
    print("\n" + "=" * 60)
    print("响应元数据")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    response = llm.invoke("你好")
    
    print("\n📊 响应元数据:\n")
    print(f"   content: {response.content}")
    print(f"   type: {response.type}")
    print(f"   id: {response.id}")
    print(f"   response_metadata: {response.response_metadata}")
    
    if hasattr(response, 'usage_metadata'):
        print(f"   usage_metadata: {response.usage_metadata}")


def main():
    demo_model_creation()
    demo_invoke()
    demo_batch()
    demo_stream()
    
    # 运行异步演示
    asyncio.run(demo_async())
    
    demo_model_parameters()
    demo_response_metadata()
    
    print("\n" + "=" * 60)
    print("模型接口总结")
    print("=" * 60)
    print("""
┌─────────────────┬─────────────────────────────────────────┐
│ 调用方式         │ 用途                                    │
├─────────────────┼─────────────────────────────────────────┤
│ invoke()        │ 同步单条调用                            │
│ ainvoke()       │ 异步单条调用                            │
│ batch()         │ 同步批量调用                            │
│ abatch()        │ 异步批量调用                            │
│ stream()        │ 同步流式调用                            │
│ astream()       │ 异步流式调用                            │
└─────────────────┴─────────────────────────────────────────┘

💡 最佳实践:
1. 简单场景使用 invoke()
2. 批量处理使用 batch() 提高效率
3. 实时响应使用 stream()
4. 高并发场景使用 ainvoke() + asyncio
    """)
    
    print("\n✅ 模型接口学习完成！")


if __name__ == "__main__":
    main()
