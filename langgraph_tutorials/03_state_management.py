#!/usr/bin/env python3
"""
LangGraph 核心概念 03 - 状态管理
功能：理解状态的传递、更新和 Reducer 机制

核心概念：
- Reducer: 控制状态如何更新的函数
- 状态传递: 节点间状态的流动
- 状态隔离: 每个步骤的状态独立性
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
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


# ========== 1. 定义带 Reducer 的状态 ==========
class ChatState(TypedDict):
    """
    带 Reducer 的状态定义
    
    Annotated[type, reducer] 语法指定如何更新该字段
    - add: 列表追加（而不是覆盖）
    """
    messages: Annotated[List, add]  # 使用 add reducer，消息会追加
    user_input: str                 # 普通字段，会被覆盖
    context: str                    # 上下文信息
    iteration: int                  # 迭代计数


# ========== 2. 定义节点 ==========
def user_input_node(state: ChatState) -> ChatState:
    """
    接收用户输入节点
    """
    print(f"👤 [User] 输入: {state['user_input']}")
    
    # 添加用户消息到历史
    return {
        **state,
        "messages": [HumanMessage(content=state['user_input'])],
        "iteration": state.get("iteration", 0) + 1
    }


def llm_response_node(state: ChatState) -> ChatState:
    """
    LLM 响应节点
    """
    print(f"🤖 [LLM] 生成响应... (迭代 {state['iteration']})")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 使用消息历史作为上下文
    messages = [
        SystemMessage(content="你是一个有帮助的助手。保持回答简洁。")
    ] + state["messages"]
    
    response = llm.invoke(messages)
    
    return {
        **state,
        "messages": [AIMessage(content=response.content)]
    }


def reflection_node(state: ChatState) -> ChatState:
    """
    反思节点：评估回答质量
    """
    print("🤔 [Reflection] 评估回答...")
    
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    
    if last_ai_message:
        reflection = f"已生成回答，长度: {len(last_ai_message.content)} 字符"
        return {
            **state,
            "context": reflection
        }
    return state


# ========== 3. 构建图 ==========
def build_chat_graph():
    """构建带状态管理的对话图"""
    
    workflow = StateGraph(ChatState)
    
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("llm_response", llm_response_node)
    workflow.add_node("reflection", reflection_node)
    
    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", "llm_response")
    workflow.add_edge("llm_response", "reflection")
    workflow.add_edge("reflection", END)
    
    return workflow.compile()


# ========== 4. 多轮对话测试 ==========
def test_conversation():
    """测试多轮对话，观察状态变化"""
    print("=" * 60)
    print("LangGraph 核心概念 03 - 状态管理")
    print("=" * 60)
    
    graph = build_chat_graph()
    
    # 模拟多轮对话
    conversation = [
        "你好，请介绍一下 LangGraph",
        "它和普通程序有什么区别？",
        "谢谢，我明白了"
    ]
    
    # 保存完整的消息历史
    all_messages = []
    
    for i, user_msg in enumerate(conversation, 1):
        print(f"\n{'='*50}")
        print(f"🔄 第 {i} 轮对话")
        print("="*50)
        
        # 准备状态，包含历史消息
        state = {
            "messages": all_messages.copy(),  # 传递历史
            "user_input": user_msg,
            "context": "",
            "iteration": i
        }
        
        print(f"📥 输入状态消息数: {len(state['messages'])}")
        
        # 执行图
        final_state = graph.invoke(state)
        
        # 更新历史
        all_messages = final_state["messages"]
        
        print(f"📤 输出状态消息数: {len(final_state['messages'])}")
        print(f"📊 上下文: {final_state['context']}")
        
        # 显示最后一条 AI 回复
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"💬 AI: {msg.content[:80]}...")
                break
    
    print(f"\n{'='*50}")
    print("📜 完整对话历史:")
    print("="*50)
    for msg in all_messages:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"{role} {msg.content[:60]}...")


# ========== 5. 展示 Reducer 效果 ==========
def demonstrate_reducer():
    """演示 Reducer 的工作原理"""
    print("\n" + "=" * 60)
    print("Reducer 机制演示")
    print("=" * 60)
    
    # 模拟 reducer 行为
    from operator import add
    
    old_messages = ["消息1", "消息2"]
    new_messages = ["消息3"]
    
    # 使用 add reducer（追加）
    result_add = add(old_messages, new_messages)
    print(f"\n使用 add reducer:")
    print(f"  旧: {old_messages}")
    print(f"  新: {new_messages}")
    print(f"  结果: {result_add}")
    
    # 不使用 reducer（覆盖）
    result_override = new_messages
    print(f"\n不使用 reducer（覆盖）:")
    print(f"  旧: {old_messages}")
    print(f"  新: {new_messages}")
    print(f"  结果: {result_override}")


if __name__ == "__main__":
    test_conversation()
    demonstrate_reducer()
    print("\n✅ 状态管理测试完成！")
