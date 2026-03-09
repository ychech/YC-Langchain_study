#!/usr/bin/env python3
"""
LangChain Core 核心概念 06 - LCEL 基础
功能：掌握 LangChain Expression Language 基础用法

核心概念：
- LCEL: LangChain 表达式语言
- | 操作符: 链式组合
- 输入输出类型自动推断
- 并行和流式支持
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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


def demo_basic_chain():
    """演示基础链式调用"""
    print("=" * 60)
    print("LangChain Core 核心概念 06 - LCEL 基础")
    print("=" * 60)
    
    print("\n🔗 基础链式调用 (Prompt | Model | Parser)\n")
    print("-" * 50)
    
    # 创建组件
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}。"),
        ("human", "{question}")
    ])
    
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    parser = StrOutputParser()
    
    # 使用 | 操作符组合
    chain = prompt | model | parser
    
    print("\n1️⃣ 链结构:")
    print("   prompt | model | parser")
    
    print("\n2️⃣ 调用链:")
    result = chain.invoke({
        "role": "Python 专家",
        "question": "什么是装饰器？"
    })
    
    print(f"   输入: {{'role': 'Python 专家', 'question': '什么是装饰器？'}}")
    print(f"   输出: {result[:80]}...")


def demo_input_output_types():
    """演示输入输出类型"""
    print("\n" + "=" * 60)
    print("LCEL 输入输出类型")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n📊 类型自动转换:\n")
    
    # 字符串输入
    print("1️⃣ 字符串输入 -> 字符串输出:")
    chain1 = llm | StrOutputParser()
    result1 = chain1.invoke("你好")
    print(f"   输入类型: str")
    print(f"   输出类型: {type(result1).__name__}")
    print(f"   输出: {result1[:30]}...")
    
    # 消息列表输入
    from langchain_core.messages import HumanMessage, SystemMessage
    
    print("\n2️⃣ 消息列表输入 -> 字符串输出:")
    chain2 = llm | StrOutputParser()
    result2 = chain2.invoke([
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好")
    ])
    print(f"   输入类型: list")
    print(f"   输出类型: {type(result2).__name__}")
    print(f"   输出: {result2[:30]}...")
    
    # 字典输入
    print("\n3️⃣ 字典输入 -> 字符串输出:")
    prompt = ChatPromptTemplate.from_template("告诉我关于 {topic} 的事")
    chain3 = prompt | llm | StrOutputParser()
    result3 = chain3.invoke({"topic": "AI"})
    print(f"   输入类型: dict")
    print(f"   输出类型: {type(result3).__name__}")
    print(f"   输出: {result3[:30]}...")


def demo_branching():
    """演示分支逻辑"""
    print("\n" + "=" * 60)
    print("LCEL 分支逻辑")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n🌿 使用 RunnableParallel 创建分支:\n")
    
    # 创建并行分支
    chain = RunnableParallel(
        summary=ChatPromptTemplate.from_template("总结以下内容: {text}") | llm | StrOutputParser(),
        keywords=ChatPromptTemplate.from_template("提取关键词: {text}") | llm | StrOutputParser(),
        sentiment=ChatPromptTemplate.from_template("分析情感: {text}") | llm | StrOutputParser()
    )
    
    text = "Python 是一门优秀的编程语言，简洁优雅，功能强大。"
    
    print(f"   输入: '{text[:30]}...'")
    print(f"   并行处理: 总结 + 关键词 + 情感\n")
    
    result = chain.invoke({"text": text})
    
    print("   输出:")
    for key, value in result.items():
        print(f"     {key}: {value[:50]}...")


def demo_nested_chains():
    """演示嵌套链"""
    print("\n" + "=" * 60)
    print("LCEL 嵌套链")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n🏗️ 构建嵌套链:\n")
    
    # 子链 1: 生成大纲
    outline_chain = (
        ChatPromptTemplate.from_template("为 '{topic}' 生成大纲")
        | llm
        | StrOutputParser()
    )
    
    # 子链 2: 生成内容
    content_chain = (
        ChatPromptTemplate.from_template("基于以下大纲写详细内容:\n{outline}")
        | llm
        | StrOutputParser()
    )
    
    # 主链: 大纲 -> 内容
    main_chain = (
        RunnablePassthrough.assign(outline=outline_chain)
        | content_chain
    )
    
    print("   链结构:")
    print("   1. outline_chain: topic -> outline")
    print("   2. content_chain: outline -> content")
    print("   3. main_chain: topic -> outline -> content\n")
    
    result = main_chain.invoke({"topic": "人工智能"})
    print(f"   输出: {result[:100]}...")


def demo_chain_config():
    """演示链配置"""
    print("\n" + "=" * 60)
    print("LCEL 链配置")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n⚙️ 使用 config 配置链:\n")
    
    chain = (
        ChatPromptTemplate.from_template("回答: {question}")
        | llm
        | StrOutputParser()
    )
    
    # 带配置的调用
    result = chain.invoke(
        {"question": "什么是 LangChain？"},
        config={
            "run_name": "my_chain_run",
            "tags": ["demo", "lcel"],
            "metadata": {"user_id": "123"}
        }
    )
    
    print(f"   配置: run_name='my_chain_run', tags=['demo', 'lcel']")
    print(f"   输出: {result[:50]}...")


def main():
    demo_basic_chain()
    demo_input_output_types()
    demo_branching()
    demo_nested_chains()
    demo_chain_config()
    
    print("\n" + "=" * 60)
    print("LCEL 基础总结")
    print("=" * 60)
    print("""
┌─────────────────┬─────────────────────────────────────────┐
│ LCEL 特性        │ 说明                                    │
├─────────────────┼─────────────────────────────────────────┤
│ | 操作符         │ 组合组件为链                            │
│ 类型推断         │ 自动处理输入输出类型转换                 │
│ 流式支持         │ 自动支持 stream() 方法                  │
│ 异步支持         │ 自动支持 ainvoke() 方法                 │
│ 批处理           │ 自动支持 batch() 方法                   │
└─────────────────┴─────────────────────────────────────────┘

💡 最佳实践:
1. 使用 | 操作符组合组件
2. 利用类型自动转换简化代码
3. 使用 RunnableParallel 并行处理
4. 使用 RunnablePassthrough 传递数据
5. 使用 config 添加元数据
    """)
    
    print("\n✅ LCEL 基础学习完成！")


if __name__ == "__main__":
    main()
