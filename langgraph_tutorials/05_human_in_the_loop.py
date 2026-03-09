#!/usr/bin/env python3
"""
LangGraph 核心概念 05 - 人机协同（Human-in-the-loop）
功能：在关键节点暂停，等待人工审核和干预

核心概念：
- interrupt: 中断执行，等待人工输入
- Command: 人工干预命令
- 审核点: 在关键决策前暂停
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
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
class HITLState(TypedDict):
    """人机协同状态"""
    messages: Annotated[List, add]
    pending_action: str  # 待审核的操作
    approved: bool       # 是否已批准
    final_output: str


# ========== 2. 定义节点 ==========
def generate_proposal(state: HITLState) -> HITLState:
    """生成提案节点"""
    print("🤖 [Generate] AI 生成提案...")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 根据用户请求生成提案
    user_request = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_request = msg.content
    
    prompt = f"用户请求: {user_request}\n\n请生成一个详细的执行方案:"
    response = llm.invoke(prompt)
    
    proposal = response.content
    print(f"\n📋 生成的提案:\n{proposal[:200]}...\n")
    
    return {
        **state,
        "pending_action": proposal,
        "messages": [AIMessage(content=f"提案: {proposal[:100]}...")]
    }


def human_review(state: HITLState) -> HITLState:
    """
    人工审核节点
    在实际应用中，这里会中断等待人工输入
    """
    print("👤 [Human Review] 等待人工审核...")
    print("-" * 50)
    
    proposal = state.get("pending_action", "")
    print(f"待审核提案:\n{proposal[:300]}...")
    print("-" * 50)
    
    # 模拟人工审核（实际应用中会真正暂停等待输入）
    # 这里为了演示，自动批准
    print("\n💡 模拟审核: 用户批准了提案")
    
    return {
        **state,
        "approved": True,
        "messages": [HumanMessage(content="批准执行")]
    }


def execute_action(state: HITLState) -> HITLState:
    """执行批准的提案"""
    print("⚡ [Execute] 执行批准的提案...")
    
    if not state.get("approved", False):
        return {
            **state,
            "final_output": "操作被拒绝"
        }
    
    # 模拟执行
    proposal = state.get("pending_action", "")
    result = f"已成功执行提案。执行内容摘要: {proposal[:100]}..."
    
    return {
        **state,
        "final_output": result,
        "messages": [AIMessage(content=result)]
    }


def reject_action(state: HITLState) -> HITLState:
    """处理拒绝的情况"""
    print("❌ [Reject] 提案被拒绝")
    return {
        **state,
        "final_output": "提案已被人工拒绝",
        "messages": [AIMessage(content="提案被拒绝")]
    }


# ========== 3. 条件路由 ==========
def route_by_approval(state: HITLState) -> str:
    """根据审核结果路由"""
    if state.get("approved", False):
        print("🛣️  [Route] -> 执行")
        return "execute"
    else:
        print("🛣️  [Route] -> 拒绝")
        return "reject"


# ========== 4. 构建图 ==========
def build_hitl_graph():
    """构建人机协同图"""
    
    workflow = StateGraph(HITLState)
    
    workflow.add_node("generate", generate_proposal)
    workflow.add_node("review", human_review)
    workflow.add_node("execute", execute_action)
    workflow.add_node("reject", reject_action)
    
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "review")
    
    # 根据审核结果分支
    workflow.add_conditional_edges(
        "review",
        route_by_approval,
        {
            "execute": "execute",
            "reject": "reject"
        }
    )
    
    workflow.add_edge("execute", END)
    workflow.add_edge("reject", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ========== 5. 基础演示 ==========
def demo_basic_hitl():
    """基础人机协同演示"""
    print("=" * 60)
    print("LangGraph 核心概念 05 - 人机协同")
    print("=" * 60)
    
    graph = build_hitl_graph()
    
    config = {"configurable": {"thread_id": "hitl_demo"}}
    
    print("\n📝 场景: AI 生成方案 -> 人工审核 -> 执行/拒绝\n")
    
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="帮我制定一个学习计划")],
            "pending_action": "",
            "approved": False,
            "final_output": ""
        },
        config=config
    )
    
    print(f"\n📤 最终结果: {state['final_output'][:100]}...")


# ========== 6. 交互式审核演示 ==========
def demo_interactive_review():
    """
    交互式审核演示
    展示如何在实际应用中断并等待人工输入
    """
    print("\n" + "=" * 60)
    print("交互式审核演示")
    print("=" * 60)
    
    # 简化的流程演示
    print("\n📋 人机协同流程:")
    print("-" * 50)
    print("1️⃣  AI 生成提案")
    print("2️⃣  系统暂停，等待人工审核")
    print("3️⃣  人工查看提案并决定: 批准 / 拒绝 / 修改")
    print("4️⃣  根据决定继续执行")
    
    print("\n💡 实际实现方式:")
    print("-" * 50)
    print("""
# 在实际应用中，可以使用以下方式中断:

from langgraph.types import interrupt

def human_review_node(state):
    # 中断执行，等待人工输入
    decision = interrupt({
        "proposal": state["pending_action"],
        "question": "是否批准此提案？"
    })
    
    # 恢复后继续
    return {
        **state,
        "approved": decision == "approve"
    }

# 客户端恢复执行:
graph.invoke(
    Command(resume="approve"),  # 或 "reject"
    config=config
)
    """)


# ========== 7. 常见应用场景 ==========
def show_use_cases():
    """展示人机协同的常见应用场景"""
    print("\n" + "=" * 60)
    print("人机协同应用场景")
    print("=" * 60)
    
    use_cases = [
        {
            "场景": "内容审核",
            "描述": "AI 生成内容后，人工审核后再发布",
            "示例": "社交媒体帖子、营销文案"
        },
        {
            "场景": "敏感操作确认",
            "描述": "涉及资金、隐私的操作需要人工确认",
            "示例": "转账、删除数据、修改配置"
        },
        {
            "场景": "复杂决策",
            "描述": "AI 提供建议，人工做最终决策",
            "示例": "医疗诊断建议、投资决策"
        },
        {
            "场景": "质量把关",
            "描述": "AI 生成代码/方案，人工审核质量",
            "示例": "代码审查、合同审核"
        },
        {
            "场景": "异常处理",
            "描述": "遇到异常情况时暂停，人工介入",
            "示例": "客服机器人转人工"
        }
    ]
    
    for i, case in enumerate(use_cases, 1):
        print(f"\n{i}. {case['场景']}")
        print(f"   描述: {case['描述']}")
        print(f"   示例: {case['示例']}")


if __name__ == "__main__":
    demo_basic_hitl()
    demo_interactive_review()
    show_use_cases()
    print("\n✅ 人机协同测试完成！")
