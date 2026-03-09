#!/usr/bin/env python3
"""
LangGraph 核心概念 01 - StateGraph 基础
功能：构建最简单的图，理解 State、Node、Edge 的基本概念

核心概念：
- State: 状态，整个图的共享数据容器
- Node: 节点，执行任务的函数
- Edge: 边，连接节点控制流程
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
load_dotenv()


# ========== 1. 定义状态 ==========
class AgentState(TypedDict):
    """
    定义图的状态结构
    所有节点都可以读取和修改这个状态
    """
    input: str           # 用户输入
    output: str          # 最终输出
    step_count: int      # 步骤计数


# ========== 2. 定义节点函数 ==========
# 起始节点
def node_start(state: AgentState) -> AgentState:
    """
    起始节点：初始化处理
    
    参数 state: 当前状态
    返回: 更新后的状态
    """
    print(f"🚀 [Start] 收到输入: {state['input']}")
    return {
        **state,
        "step_count": 1
    }

# 处理节点
def node_process(state: AgentState) -> AgentState:
    """
    处理节点：调用 LLM 处理
    """
    print(f"⚙️  [Process] 正在处理... (步骤 {state['step_count']})")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    )
    
    prompt = f"请用一句话回答：{state['input']}"
    response = llm.invoke(prompt)
    
    return {
        **state,
        "output": response.content,
        "step_count": state["step_count"] + 1
    }

# 结束节点
def node_end(state: AgentState) -> AgentState:
    """
    结束节点：收尾工作
    """
    print(f"✅ [End] 处理完成！总步骤: {state['step_count']}")
    print(f"📤 最终输出: {state['output'][:100]}...")
    return state


# ========== 3. 构建图 ==========
def build_graph():
    """构建 StateGraph"""
    
    # 创建图，指定状态类型
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("start", node_start)
    workflow.add_node("process", node_process)
    workflow.add_node("end", node_end)
    
    # 添加边（连接节点）
    workflow.add_edge("start", "process")  # start -> process
    workflow.add_edge("process", "end")    # process -> end
    workflow.add_edge("end", END)          # end -> 结束
    
    # 设置入口点
    workflow.set_entry_point("start")
    
    # 编译图
    return workflow.compile()


# ========== 4. 运行图 ==========
def main():
    print("=" * 60)
    print("LangGraph 核心概念 01 - StateGraph 基础")
    print("=" * 60)
    
    # 构建图
    graph = build_graph()
    
    # 准备初始状态
    initial_state = {
        "input": "什么是 LangGraph？",
        "output": "",
        "step_count": 0
    }
    
    print(f"\n📥 初始状态: {initial_state}\n")
    
    # 运行图
    final_state = graph.invoke(initial_state)
    
    print(f"\n📊 最终状态:")
    print(f"  - 输入: {final_state['input']}")
    print(f"  - 输出: {final_state['output']}")
    print(f"  - 步骤数: {final_state['step_count']}")
    
    print("\n✅ 图执行完成！")
    
    # 可视化图结构（保存为 PNG）
    try:
        from langgraph.graph import visualize
        # 注意：需要安装额外依赖才能可视化
        print("\n💡 提示：安装 graphviz 可以可视化图结构")
        print("   pip install graphviz")
    except:
        pass


# ========== 5. 进阶：查看执行过程 ==========
def run_with_stream():
    """流式执行，查看每个节点的变化"""
    print("\n" + "=" * 60)
    print("流式执行 - 查看状态变化")
    print("=" * 60)
    
    graph = build_graph()
    
    initial_state = {
        "input": "Python 有什么优点？",
        "output": "",
        "step_count": 0
    }
    
    # 流式执行
    for event in graph.stream(initial_state):
        print(f"\n📦 事件: {event}")


if __name__ == "__main__":
    main()
    run_with_stream()
