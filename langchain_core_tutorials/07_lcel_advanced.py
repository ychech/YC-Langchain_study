#!/usr/bin/env python3
"""
LangChain Core 核心概念 07 - LCEL 进阶
功能：掌握 LCEL 的高级用法

核心概念：
- 动态路由
- 条件逻辑
- 错误处理
- 自定义 Runnable
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda, 
    RunnablePassthrough, 
    RunnableParallel,
    RunnableBranch
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


def demo_runnable_branch():
    """演示条件分支"""
    print("=" * 60)
    print("LangChain Core 核心概念 07 - LCEL 进阶")
    print("=" * 60)
    
    print("\n🌿 RunnableBranch - 条件分支\n")
    print("-" * 50)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 定义不同场景的处理链
    math_chain = (
        ChatPromptTemplate.from_template("计算: {input}")
        | llm
        | StrOutputParser()
    )
    
    translation_chain = (
        ChatPromptTemplate.from_template("翻译: {input}")
        | llm
        | StrOutputParser()
    )
    
    general_chain = (
        ChatPromptTemplate.from_template("回答: {input}")
        | llm
        | StrOutputParser()
    )
    
    # 创建条件分支
    branch = RunnableBranch(
        # 条件 1: 如果是数学问题
        (lambda x: "计算" in x["input"] or any(c.isdigit() for c in x["input"]), math_chain),
        # 条件 2: 如果是翻译请求
        (lambda x: "翻译" in x["input"], translation_chain),
        # 默认: 通用回答
        general_chain
    )
    
    print("1️⃣ 条件分支链:")
    print("   - 包含数字/'计算' -> math_chain")
    print("   - 包含'翻译' -> translation_chain")
    print("   - 其他 -> general_chain\n")
    
    # 测试不同输入
    test_inputs = [
        {"input": "计算 2 + 2"},
        {"input": "翻译 'hello'"},
        {"input": "什么是 Python？"}
    ]
    
    for inp in test_inputs:
        result = branch.invoke(inp)
        print(f"   输入: '{inp['input']}'")
        print(f"   输出: {result[:50]}...\n")


def demo_dynamic_routing():
    """演示动态路由"""
    print("\n" + "=" * 60)
    print("动态路由")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n🛣️ 根据输入动态选择处理路径:\n")
    
    # 路由函数
    def route(info):
        topic = info["topic"].lower()
        if "python" in topic:
            return "python"
        elif "java" in topic:
            return "java"
        else:
            return "general"
    
    # 不同主题的处理链
    python_chain = (
        ChatPromptTemplate.from_template("作为 Python 专家，介绍: {topic}")
        | llm
        | StrOutputParser()
    )
    
    java_chain = (
        ChatPromptTemplate.from_template("作为 Java 专家，介绍: {topic}")
        | llm
        | StrOutputParser()
    )
    
    general_chain = (
        ChatPromptTemplate.from_template("介绍: {topic}")
        | llm
        | StrOutputParser()
    )
    
    # 创建路由链
    chain = (
        RunnablePassthrough.assign(destination=RunnableLambda(route))
        | RunnableLambda(lambda x: {
            "python": python_chain,
            "java": java_chain,
            "general": general_chain
        }[x["destination"]].invoke(x))
    )
    
    print("   路由规则:")
    print("   - topic 包含 'python' -> python_chain")
    print("   - topic 包含 'java' -> java_chain")
    print("   - 其他 -> general_chain\n")
    
    for topic in ["Python 装饰器", "Java 虚拟机", "Rust 语言"]:
        result = chain.invoke({"topic": topic})
        print(f"   主题: '{topic}'")
        print(f"   结果: {result[:60]}...\n")


def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("错误处理")
    print("=" * 60)
    
    print("\n🛡️ 在 LCEL 中处理错误:\n")
    
    # 可能出错的函数
    def risky_operation(x):
        if x["value"] == 0:
            raise ValueError("不能为 0")
        return {"result": 100 / x["value"]}
    
    # 包装错误处理
    def safe_operation(x):
        try:
            return risky_operation(x)
        except Exception as e:
            return {"error": str(e), "result": None}
    
    safe_chain = RunnableLambda(safe_operation)
    
    print("1️⃣ 使用 try-except 包装:")
    
    for val in [10, 0, 5]:
        result = safe_chain.invoke({"value": val})
        print(f"   输入: {{'value': {val}}}")
        print(f"   输出: {result}")
    
    # 使用 fallback
    print("\n2️⃣ 使用 fallback 机制:")
    
    primary_chain = RunnableLambda(lambda x: {"result": 100 / x["value"]})
    fallback_chain = RunnableLambda(lambda x: {"result": "计算失败，使用默认值: 0"})
    
    # 注意：实际的 fallback 需要用更复杂的错误处理
    # 这里演示概念
    def with_fallback(primary, fallback):
        def wrapper(x):
            try:
                return primary.invoke(x)
            except:
                return fallback.invoke(x)
        return RunnableLambda(wrapper)
    
    safe_chain2 = with_fallback(primary_chain, fallback_chain)
    
    for val in [10, 0]:
        result = safe_chain2.invoke({"value": val})
        print(f"   输入: {{'value': {val}}}")
        print(f"   输出: {result}")


def demo_custom_runnable():
    """演示自定义 Runnable"""
    print("\n" + "=" * 60)
    print("自定义 Runnable")
    print("=" * 60)
    
    from langchain_core.runnables import Runnable
    from typing import Any, Dict
    
    print("\n🔧 继承 Runnable 创建自定义组件:\n")
    
    class LoggingRunnable(Runnable):
        """带日志记录的 Runnable"""
        
        def __init__(self, name: str):
            self.name = name
        
        def invoke(self, input: Any, config: Dict = None) -> Any:
            print(f"   [LOG] {self.name} 开始处理: {input}")
            result = f"处理结果: {input}"
            print(f"   [LOG] {self.name} 完成处理")
            return result
        
        @property
        def InputType(self):
            return Any
        
        @property
        def OutputType(self):
            return str
    
    # 使用自定义 Runnable
    logger = LoggingRunnable("MyLogger")
    
    print("1️⃣ 单独调用:")
    result = logger.invoke("测试数据")
    print(f"   结果: {result}")
    
    print("\n2️⃣ 在链中使用:")
    chain = logger | RunnableLambda(lambda x: x.upper())
    result = chain.invoke("链式数据")
    print(f"   结果: {result}")


def demo_chain_optimization():
    """演示链优化技巧"""
    print("\n" + "=" * 60)
    print("链优化技巧")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n⚡ 优化技巧:\n")
    
    print("1️⃣ 使用 bind 预绑定参数:")
    
    # 预绑定 temperature
    creative_llm = llm.bind(temperature=0.9)
    conservative_llm = llm.bind(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("写一句关于 {topic} 的话")
    
    creative_chain = prompt | creative_llm | StrOutputParser()
    conservative_chain = prompt | conservative_llm | StrOutputParser()
    
    print("   creative (temp=0.9):")
    print(f"   {creative_chain.invoke({'topic': '春天'})[:40]}...")
    
    print("\n   conservative (temp=0.1):")
    print(f"   {conservative_chain.invoke({'topic': '春天'})[:40]}...")
    
    print("\n2️⃣ 使用 partial 预填充变量:")
    
    template = ChatPromptTemplate.from_messages([
        ("system", "你是 {role}。"),
        ("human", "{question}")
    ])
    
    # 预填充 role
    expert_template = template.partial(role="Python 专家")
    
    chain = expert_template | llm | StrOutputParser()
    result = chain.invoke({"question": "什么是 GIL？"})
    print(f"   预填充 role='Python 专家'")
    print(f"   结果: {result[:50]}...")


def main():
    demo_runnable_branch()
    demo_dynamic_routing()
    demo_error_handling()
    demo_custom_runnable()
    demo_chain_optimization()
    
    print("\n" + "=" * 60)
    print("LCEL 进阶总结")
    print("=" * 60)
    print("""
┌─────────────────────┬────────────────────────────────────┐
│ 高级特性             │ 说明                               │
├─────────────────────┼────────────────────────────────────┤
│ RunnableBranch      │ 条件分支，根据条件选择不同路径      │
│ 动态路由             │ 运行时决定执行路径                  │
│ 错误处理             │ try-except 包装或使用 fallback      │
│ 自定义 Runnable      │ 继承 Runnable 基类创建组件          │
│ bind()              │ 预绑定模型参数                      │
│ partial()           │ 预填充提示词变量                    │
└─────────────────────┴────────────────────────────────────┘

💡 最佳实践:
1. 使用 RunnableBranch 处理条件逻辑
2. 包装错误处理提高健壮性
3. 使用 bind() 创建变体模型
4. 使用 partial() 提高模板复用性
5. 自定义 Runnable 封装业务逻辑
    """)
    
    print("\n✅ LCEL 进阶学习完成！")


if __name__ == "__main__":
    main()
