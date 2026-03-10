#!/usr/bin/env python3
"""
LangChain Core 核心概念 08 - 回调机制 (Callbacks)
功能：掌握事件监听和回调处理

核心概念：
- BaseCallbackHandler: 回调处理器基类
- 事件类型: on_llm_start, on_llm_end, on_chain_start 等
- 自定义回调: 日志、监控、追踪
"""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
import os


# 加载环境变量
load_dotenv()


class LoggingCallbackHandler(BaseCallbackHandler):
    """日志回调处理器"""
    
    def __init__(self):
        self.start_times = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始调用"""
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()
        print(f"\n🤖 [LLM Start] 调用开始")
        print(f"   提示词数量: {len(prompts)}")
    
    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束"""
        run_id = kwargs.get("run_id")
        duration = time.time() - self.start_times.get(run_id, 0)
        print(f"\n✅ [LLM End] 调用完成")
        print(f"   耗时: {duration:.2f}s")
        if hasattr(response, 'generations'):
            print(f"   生成数量: {len(response.generations)}")
    
    def on_llm_error(self, error, **kwargs):
        """LLM 调用出错"""
        print(f"\n❌ [LLM Error] 错误: {error}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """链开始执行"""
        print(f"\n🔗 [Chain Start] 链开始执行")
        print(f"   输入: {str(inputs)[:50]}...")
    
    def on_chain_end(self, outputs, **kwargs):
        """链执行结束"""
        print(f"\n✅ [Chain End] 链执行完成")
        print(f"   输出: {str(outputs)[:50]}...")


class TokenCountCallbackHandler(BaseCallbackHandler):
    """Token 计数回调处理器"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """统计 Token 使用量"""
        # 注意：实际 token 数需要从 response_metadata 获取
        print(f"\n📊 [Token Count]")
        print(f"   本次调用已记录")


class CostTrackingCallbackHandler(BaseCallbackHandler):
    """成本追踪回调处理器"""
    
    def __init__(self):
        self.total_calls = 0
        self.total_cost = 0.0
        self.model_prices = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        }
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.total_calls += 1
        print(f"\n💰 [Cost Track] 第 {self.total_calls} 次调用")
    
    def on_llm_end(self, response, **kwargs):
        # 简化成本计算
        estimated_cost = 0.001  # 假设每次调用成本
        self.total_cost += estimated_cost
        print(f"   累计成本: ${self.total_cost:.4f}")


def demo_basic_callback():
    """演示基础回调"""
    print("=" * 60)
    print("LangChain Core 核心概念 08 - 回调机制")
    print("=" * 60)
    
    print("\n📡 基础回调演示\n")
    print("-" * 50)
    
    # 创建回调处理器
    callbacks = [LoggingCallbackHandler()]
    
    # 创建模型，传入回调
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        callbacks=callbacks
    )
    
    print("\n1️⃣ 单次调用:")
    response = llm.invoke("你好")
    print(f"\n   最终输出: {response.content[:30]}...")


def demo_chain_callback():
    """演示链的回调"""
    print("\n" + "=" * 60)
    print("链的回调")
    print("=" * 60)
    
    # 创建回调处理器
    callbacks = [LoggingCallbackHandler(), CostTrackingCallbackHandler()]
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 创建链
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是助手。"),
            ("human", "{question}")
        ])
        | llm
        | StrOutputParser()
    )
    
    print("\n🔗 链执行回调:\n")
    
    result = chain.invoke(
        {"question": "什么是回调函数？"},
        config={"callbacks": callbacks}
    )
    
    print(f"\n📤 最终结果: {result[:50]}...")


def demo_multiple_callbacks():
    """演示多个回调处理器"""
    print("\n" + "=" * 60)
    print("多个回调处理器")
    print("=" * 60)
    
    # 多个回调
    callbacks = [
        LoggingCallbackHandler(),
        TokenCountCallbackHandler(),
        CostTrackingCallbackHandler()
    ]
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n📡 多个回调同时工作:\n")
    
    chain = (
        ChatPromptTemplate.from_template("回答: {question}")
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke(
        {"question": "Python 有什么优点？"},
        config={"callbacks": callbacks}
    )
    
    print(f"\n📤 结果: {result[:50]}...")


def demo_custom_metrics():
    """演示自定义指标收集"""
    print("\n" + "=" * 60)
    print("自定义指标收集")
    print("=" * 60)
    
    class MetricsCallbackHandler(BaseCallbackHandler):
        """指标收集回调"""
        
        def __init__(self):
            self.metrics = {
                "llm_calls": 0,
                "chain_starts": 0,
                "chain_ends": 0,
                "errors": 0
            }
        
        def on_llm_start(self, serialized, prompts, **kwargs):
            self.metrics["llm_calls"] += 1
        
        def on_chain_start(self, serialized, inputs, **kwargs):
            self.metrics["chain_starts"] += 1
        
        def on_chain_end(self, outputs, **kwargs):
            self.metrics["chain_ends"] += 1
        
        def on_llm_error(self, error, **kwargs):
            self.metrics["errors"] += 1
        
        def print_metrics(self):
            print("\n📊 指标统计:")
            for key, value in self.metrics.items():
                print(f"   {key}: {value}")
    
    metrics = MetricsCallbackHandler()
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    chain = (
        ChatPromptTemplate.from_template("{task}")
        | llm
        | StrOutputParser()
    )
    
    print("\n🔄 执行多个任务:\n")
    
    tasks = [
        "什么是 AI？",
        "什么是 ML？",
        "什么是 DL？"
    ]
    
    for task in tasks:
        chain.invoke({"task": task}, config={"callbacks": [metrics]})
    
    metrics.print_metrics()


def show_callback_events():
    """展示回调事件类型"""
    print("\n" + "=" * 60)
    print("回调事件类型参考")
    print("=" * 60)
    
    events = {
        "on_llm_start": "LLM 开始生成",
        "on_llm_new_token": "LLM 生成新 token（流式）",
        "on_llm_end": "LLM 生成完成",
        "on_llm_error": "LLM 生成出错",
        "on_chain_start": "链开始执行",
        "on_chain_end": "链执行完成",
        "on_tool_start": "工具开始执行",
        "on_tool_end": "工具执行完成",
        "on_text": "任意文本输出",
    }
    
    print("\n📋 可用回调事件:\n")
    for event, desc in events.items():
        print(f"   {event:25} - {desc}")


def main():
    demo_basic_callback()
    demo_chain_callback()
    show_callback_events()
    
    print("\n" + "=" * 60)
    print("回调机制总结")
    print("=" * 60)
    print("""
┌─────────────────────────┬────────────────────────────────────┐
│ 回调处理器               │ 用途                               │
├─────────────────────────┼────────────────────────────────────┤
│ BaseCallbackHandler     │ 基类，继承实现自定义回调            │
│ LoggingCallbackHandler  │ 日志记录                           │
│ StdOutCallbackHandler   │ 标准输出（内置）                    │
│ FileCallbackHandler     │ 文件记录（内置）                    │
└─────────────────────────┴────────────────────────────────────┘

💡 最佳实践:
1. 继承 BaseCallbackHandler 创建自定义回调
2. 使用 callbacks 参数传入回调列表
3. 在 config 中传递 callbacks
4. 多个回调可以组合使用
5. 利用回调实现监控、日志、追踪

🎯 应用场景:
- 日志记录和调试
- 性能监控和指标收集
- 成本追踪
- 错误处理和告警
- 实时状态更新
    """)
    
    print("\n✅ 回调机制学习完成！")


if __name__ == "__main__":
    main()
