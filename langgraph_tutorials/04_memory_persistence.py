#!/usr/bin/env python3
"""
LangGraph 核心概念 04 - 记忆与持久化
功能：保存和恢复图的状态，实现长期记忆

核心概念：
- Checkpoint: 检查点，保存图的完整状态
- MemorySaver: 内存中的状态保存
- 线程ID: 区分不同对话会话
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from operator import add
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


# ========== 1. 定义状态 ==========
class PersistentState(TypedDict):
    """带持久化的状态"""
    messages: Annotated[List, add]
    user_name: str  # 记住用户名字
    session_info: str


# ========== 2. 定义节点 ==========
def chat_node(state: PersistentState) -> PersistentState:
    """对话节点"""
    print(f"🤖 [Chat] 生成回复...")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 构建系统提示词，包含记忆的信息
    user_name = state.get("user_name", "")
    system_content = "你是一个友好的助手。"
    if user_name:
        system_content += f"用户名字叫 {user_name}。"
    
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        **state,
        "messages": [AIMessage(content=response.content)]
    }


def extract_info_node(state: PersistentState) -> PersistentState:
    """提取用户信息节点"""
    print("🔍 [Extract] 提取用户信息...")
    
    # 简单规则：检查用户输入是否包含"我叫"
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            content = msg.content
            if "我叫" in content:
                # 提取名字（简化处理）
                name = content.split("我叫")[-1].strip().strip("！。")
                print(f"   提取到名字: {name}")
                return {**state, "user_name": name}
    
    return state


# ========== 3. 构建带持久化的图 ==========
def build_persistent_graph():
    """构建带持久化的图"""
    
    workflow = StateGraph(PersistentState)
    
    workflow.add_node("chat", chat_node)
    workflow.add_node("extract_info", extract_info_node)
    
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "chat")
    workflow.add_edge("chat", END)
    
    # 添加检查点保存器（内存中）
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# ========== 4. 测试持久化 ==========
def test_persistence():
    """测试状态持久化"""
    print("=" * 60)
    print("LangGraph 核心概念 04 - 记忆与持久化")
    print("=" * 60)
    
    graph = build_persistent_graph()
    
    # 配置：指定线程ID（会话ID）
    config = {
        "configurable": {
            "thread_id": "session_001"  # 会话ID
        }
    }
    
    print("\n📝 第一轮对话：自我介绍")
    print("-" * 50)
    
    state1 = graph.invoke(
        {
            "messages": [HumanMessage(content="你好，我叫张三")],
            "user_name": "",
            "session_info": ""
        },
        config=config
    )
    
    print(f"AI: {state1['messages'][-1].content[:100]}...")
    print(f"记住的名字: {state1['user_name']}")
    
    # 继续对话（使用相同的 thread_id）
    print("\n📝 第二轮对话：询问名字")
    print("-" * 50)
    
    state2 = graph.invoke(
        {
            "messages": [HumanMessage(content="我叫什么名字？")],
            "user_name": state1["user_name"],  # 传递之前的状态
            "session_info": ""
        },
        config=config
    )
    
    print(f"AI: {state2['messages'][-1].content[:100]}...")
    
    # 检查状态是否保存
    print("\n💾 检查保存的状态:")
    print("-" * 50)
    
    # 获取当前状态
    current_state = graph.get_state(config)
    if current_state:
        print(f"找到保存的状态!")
        print(f"  - 用户名字: {current_state.values.get('user_name', '无')}")
        print(f"  - 消息数量: {len(current_state.values.get('messages', []))}")
    
    return graph, config


# ========== 5. 多会话测试 ==========
def test_multi_session():
    """测试多个独立会话"""
    print("\n" + "=" * 60)
    print("多会话测试")
    print("=" * 60)
    
    graph = build_persistent_graph()
    
    # 会话1：张三
    config1 = {"configurable": {"thread_id": "user_zhangsan"}}
    state1 = graph.invoke(
        {"messages": [HumanMessage(content="我叫张三")], "user_name": "", "session_info": ""},
        config=config1
    )
    
    # 会话2：李四
    config2 = {"configurable": {"thread_id": "user_lisi"}}
    state2 = graph.invoke(
        {"messages": [HumanMessage(content="我叫李四")], "user_name": "", "session_info": ""},
        config=config2
    )
    
    print("\n会话1 (张三):")
    print(f"  记住的名字: {state1['user_name']}")
    
    print("\n会话2 (李四):")
    print(f"  记住的名字: {state2['user_name']}")
    
    # 验证状态隔离
    print("\n✅ 会话状态隔离验证:")
    state1_saved = graph.get_state(config1)
    state2_saved = graph.get_state(config2)
    
    print(f"  会话1保存: {state1_saved.values.get('user_name') if state1_saved else 'None'}")
    print(f"  会话2保存: {state2_saved.values.get('user_name') if state2_saved else 'None'}")


# ========== 6. 状态历史查看 ==========
def show_state_history():
    """展示状态历史"""
    print("\n" + "=" * 60)
    print("查看状态历史")
    print("=" * 60)
    
    graph = build_persistent_graph()
    config = {"configurable": {"thread_id": "history_demo"}}
    
    # 进行多轮对话
    for i, msg in enumerate(["你好", "今天天气怎么样？", "谢谢"], 1):
        print(f"\n第 {i} 轮:")
        state = graph.invoke(
            {"messages": [HumanMessage(content=msg)], "user_name": "", "session_info": ""},
            config=config
        )
        print(f"  消息数: {len(state['messages'])}")
    
    # 查看所有历史状态
    print("\n📜 状态历史:")
    print("-" * 50)
    
    try:
        history = list(graph.get_state_history(config))
        for i, state in enumerate(history[:5], 1):
            print(f"  {i}. 消息数: {len(state.values.get('messages', []))}")
    except Exception as e:
        print(f"获取历史需要更多配置: {e}")


if __name__ == "__main__":
    test_persistence()
    test_multi_session()
    show_state_history()
    print("\n✅ 记忆与持久化测试完成！")
