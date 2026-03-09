#!/usr/bin/env python3
"""
LangGraph 核心概念 02 - 条件边（Conditional Edges）
功能：实现 if/else 分支逻辑，根据状态决定流程走向

核心概念：
- 条件边：根据状态决定下一个节点
- 路由函数：返回下一个节点的名称
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
load_dotenv()


# ========== 1. 定义状态 ==========
class RouterState(TypedDict):
    """带路由信息的状态"""
    input: str           # 用户输入
    intent: str          # 识别出的意图
    output: str          # 输出结果


# ========== 2. 定义节点 ==========
def classify_intent(state: RouterState) -> RouterState:
    """
    意图识别节点：判断用户意图类型
    """
    print(f"🔍 [Classify] 分析输入: {state['input']}")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.1
    )
    
    # 让 LLM 判断意图
    prompt = f"""判断以下用户输入的意图类型，只能返回以下之一：
- "weather": 询问天气
- "math": 数学计算
- "chat": 闲聊/其他

用户输入: {state['input']}

意图:"""
    
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    
    # 标准化意图
    if "weather" in intent:
        intent = "weather"
    elif "math" in intent:
        intent = "math"
    else:
        intent = "chat"
    
    print(f"   识别意图: {intent}")
    return {**state, "intent": intent}


def handle_weather(state: RouterState) -> RouterState:
    """处理天气查询"""
    print("☀️  [Weather] 处理天气查询")
    return {
        **state,
        "output": "今天天气晴朗，25°C，适合外出！"
    }


def handle_math(state: RouterState) -> RouterState:
    """处理数学计算"""
    print("🔢 [Math] 处理数学计算")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    prompt = f"计算以下表达式，只返回结果:\n{state['input']}"
    response = llm.invoke(prompt)
    
    return {
        **state,
        "output": f"计算结果: {response.content}"
    }


def handle_chat(state: RouterState) -> RouterState:
    """处理闲聊"""
    print("💬 [Chat] 处理闲聊")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    response = llm.invoke(state['input'])
    
    return {
        **state,
        "output": response.content
    }


# ========== 3. 定义路由函数 ==========
def route_by_intent(state: RouterState) -> Literal["weather", "math", "chat"]:
    """
    路由函数：根据意图决定下一个节点
    
    返回: 下一个节点的名称
    """
    intent = state.get("intent", "chat")
    print(f"🛣️  [Route] 路由到: {intent}")
    return intent


# ========== 4. 构建带条件边的图 ==========
def build_router_graph():
    """构建带路由的图"""
    
    workflow = StateGraph(RouterState)
    
    # 添加节点
    workflow.add_node("classify", classify_intent)
    workflow.add_node("weather", handle_weather)
    workflow.add_node("math", handle_math)
    workflow.add_node("chat", handle_chat)
    
    # 设置入口
    workflow.set_entry_point("classify")
    
    # 添加条件边：从 classify 节点，根据意图路由到不同节点
    workflow.add_conditional_edges(
        "classify",           # 起始节点
        route_by_intent,      # 路由函数
        {                     # 路由映射
            "weather": "weather",
            "math": "math",
            "chat": "chat"
        }
    )
    
    # 所有处理节点都连接到结束
    workflow.add_edge("weather", END)
    workflow.add_edge("math", END)
    workflow.add_edge("chat", END)
    
    return workflow.compile()


# ========== 5. 运行测试 ==========
def main():
    print("=" * 60)
    print("LangGraph 核心概念 02 - 条件边（Conditional Edges）")
    print("=" * 60)
    
    graph = build_router_graph()
    
    # 测试不同意图的输入
    test_inputs = [
        "北京今天天气怎么样？",           # weather
        "计算 123 乘以 456",              # math
        "你好，请介绍一下自己",            # chat
        "3 + 5 等于几？",                 # math
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"📝 用户输入: {user_input}")
        print("="*50)
        
        initial_state = {
            "input": user_input,
            "intent": "",
            "output": ""
        }
        
        final_state = graph.invoke(initial_state)
        
        print(f"\n📤 最终输出: {final_state['output'][:100]}...")
        print(f"   识别意图: {final_state['intent']}")
    
    print("\n✅ 条件边测试完成！")


if __name__ == "__main__":
    main()
