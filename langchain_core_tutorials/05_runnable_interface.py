#!/usr/bin/env python3
"""
LangChain Core 核心概念 05 - Runnable 接口
功能：理解 LangChain 的核心抽象接口

核心概念：
- Runnable: 所有可运行对象的基类
- invoke: 同步执行
- ainvoke: 异步执行
- batch: 批量执行
- stream: 流式执行
"""
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


def demo_runnable_lambda():
    """演示 RunnableLambda"""
    print("=" * 60)
    print("LangChain Core 核心概念 05 - Runnable 接口")
    print("=" * 60)
    
    print("\n🔧 RunnableLambda - 将函数包装为 Runnable\n")
    print("-" * 50)
    
    # 普通函数
    def add_one(x: int) -> int:
        return x + 1
    
    # 包装为 Runnable
    runnable = RunnableLambda(add_one)
    
    print("\n1️⃣ 基本使用:")
    print(f"   函数: add_one(x) = x + 1")
    print(f"   输入: 5")
    result = runnable.invoke(5)
    print(f"   输出: {result}")
    
    # 批量执行
    print("\n2️⃣ 批量执行:")
    results = runnable.batch([1, 2, 3, 4, 5])
    print(f"   输入: [1, 2, 3, 4, 5]")
    print(f"   输出: {results}")
    
    # 流式执行（stream 处理单个输入，逐个产出结果）
    print("\n3️⃣ 流式执行:")
    print(f"   输入: 100")
    print(f"   输出: ", end="")
    for chunk in runnable.stream(100):
        print(f"{chunk} ", end="")
    print()


def demo_runnable_passthrough():
    """演示 RunnablePassthrough"""
    print("\n" + "=" * 60)
    print("RunnablePassthrough - 透传输入")
    print("=" * 60)
    
    print("\n📍 用途: 在链中传递原始输入\n")
    
    # 基础用法
    passthrough = RunnablePassthrough()
    
    print("1️⃣ 基础用法:")
    result = passthrough.invoke("hello")
    print(f"   输入: 'hello'")
    print(f"   输出: '{result}'")
    print(f"   说明: 原样返回输入")
    
    # 在链中使用（assign 需要字典输入）
    print("\n2️⃣ 在 LCEL 链中使用:")
    
    chain = RunnablePassthrough.assign(
        upper=lambda x: x["text"].upper(),
        length=lambda x: len(x["text"])
    )
    
    result = chain.invoke({"text": "hello"})
    print(f"   输入: {{'text': 'hello'}}")
    print(f"   输出: {result}")
    print(f"   说明: 保留原始输入，并添加新字段")


def demo_runnable_parallel():
    """演示 RunnableParallel"""
    print("\n" + "=" * 60)
    print("RunnableParallel - 并行执行")
    print("=" * 60)
    
    print("\n⚡ 同时执行多个 Runnable\n")
    
    # 创建并行链
    parallel = RunnableParallel(
        upper=RunnableLambda(lambda x: x.upper()),
        lower=RunnableLambda(lambda x: x.lower()),
        length=RunnableLambda(lambda x: len(x)),
        reversed=RunnableLambda(lambda x: x[::-1])
    )
    
    print("1️⃣ 并行处理:")
    result = parallel.invoke("Hello World")
    print(f"   输入: 'Hello World'")
    print(f"   输出:")
    for key, value in result.items():
        print(f"     {key}: {value}")
    
    # 批量并行
    print("\n2️⃣ 批量并行:")
    results = parallel.batch(["ABC", "xyz"])
    for i, r in enumerate(results, 1):
        print(f"   输入 {i}: {r}")


def demo_runnable_chain():
    """演示 Runnable 链式调用"""
    print("\n" + "=" * 60)
    print("Runnable 链式调用")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n🔗 构建复杂链:\n")
    
    # 步骤 1: 预处理
    def preprocess(x: dict) -> dict:
        return {
            **x,
            "enhanced_topic": f"详细介绍: {x['topic']}"
        }
    
    # 步骤 2: 生成提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专家。"),
        ("human", "{enhanced_topic}")
    ])
    
    # 步骤 3: 解析输出
    parser = StrOutputParser()
    
    # 组合链
    chain = (
        RunnableLambda(preprocess)
        | prompt
        | llm
        | parser
    )
    
    print("   链结构: preprocess | prompt | llm | parser")
    print("   输入: {'topic': 'Python'}\n")
    
    result = chain.invoke({"topic": "Python"})
    print(f"   输出: {result[:100]}...")


def demo_runnable_interface():
    """演示 Runnable 标准接口"""
    print("\n" + "=" * 60)
    print("Runnable 标准接口方法")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n📋 所有 Runnable 都实现了以下接口:\n")
    
    # 1. invoke
    print("1️⃣ invoke - 同步单条调用")
    result = llm.invoke("你好")
    print(f"   llm.invoke('你好') -> {result.content[:20]}...")
    
    # 2. batch
    print("\n2️⃣ batch - 批量调用")
    results = llm.batch(["什么是 AI？", "什么是 ML？"])
    print(f"   llm.batch([...]) -> [{len(results)} 个结果]")
    
    # 3. stream
    print("\n3️⃣ stream - 流式调用")
    print(f"   llm.stream('你好') -> ", end="")
    for chunk in llm.stream("你好"):
        print(f"{chunk.content}", end="")
    print()
    
    # 4. ainvoke (异步)
    print("\n4️⃣ ainvoke - 异步调用")
    print(f"   llm.ainvoke('你好') -> 异步版本")


def main():
    demo_runnable_lambda()
    demo_runnable_passthrough()
    demo_runnable_parallel()
    demo_runnable_chain()
    demo_runnable_interface()
    
    print("\n" + "=" * 60)
    print("Runnable 接口总结")
    print("=" * 60)
    print("""
┌─────────────────────┬────────────────────────────────────┐
│ Runnable 类型        │ 用途                               │
├─────────────────────┼────────────────────────────────────┤
│ RunnableLambda      │ 包装函数为 Runnable                │
│ RunnablePassthrough │ 透传或扩展输入                     │
│ RunnableParallel    │ 并行执行多个 Runnable              │
│ RunnableSequence    │ 顺序执行 (由 | 操作符创建)         │
└─────────────────────┴────────────────────────────────────┘

┌─────────────────────┬────────────────────────────────────┐
│ 标准接口方法         │ 说明                               │
├─────────────────────┼────────────────────────────────────┤
│ invoke()            │ 同步单条执行                       │
│ batch()             │ 批量执行                           │
│ stream()            │ 流式执行                           │
│ ainvoke()           │ 异步执行                           │
└─────────────────────┴────────────────────────────────────┘

💡 最佳实践:
1. 使用 | 操作符构建链
2. RunnableLambda 包装自定义函数
3. RunnablePassthrough 保留原始输入
4. RunnableParallel 并行处理
    """)
    
    print("\n✅ Runnable 接口学习完成！")


if __name__ == "__main__":
    main()
