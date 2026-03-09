#!/usr/bin/env python3
"""
LangGraph 核心概念 06 - 多 Agent 系统
功能：多个专业 Agent 协作完成任务

核心概念：
- 子图: 每个 Agent 是一个独立的图
- 路由: 任务分配给不同的 Agent
- 协调: Agent 之间的通信和协作
"""
from typing import TypedDict, Annotated, List, Literal
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
class MultiAgentState(TypedDict):
    """多 Agent 系统状态"""
    messages: Annotated[List, add]
    current_agent: str      # 当前活跃的 Agent
    task_type: str          # 任务类型
    research_result: str    # 研究员结果
    writer_result: str      # 写手结果
    reviewer_result: str    # 审核员结果
    final_output: str       # 最终输出


# ========== 2. 定义专业 Agent ==========
class ResearchAgent:
    """研究 Agent：负责信息收集和研究"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )
    
    def run(self, topic: str) -> str:
        """执行研究任务"""
        print(f"🔬 [ResearchAgent] 研究主题: {topic}")
        
        prompt = f"""作为研究员，请对以下主题进行深入研究，提供关键信息、背景和重要观点：

主题: {topic}

请提供：
1. 核心概念解释
2. 主要应用场景
3. 优缺点分析
4. 发展趋势

研究报告:"""
        
        response = self.llm.invoke(prompt)
        return response.content


class WriterAgent:
    """写作 Agent：负责内容创作"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.8
        )
    
    def run(self, topic: str, research: str) -> str:
        """执行写作任务"""
        print(f"✍️  [WriterAgent] 撰写内容: {topic}")
        
        prompt = f"""作为专业写手，请基于以下研究资料撰写一篇高质量文章。

主题: {topic}

研究资料:
{research}

要求：
- 结构清晰，有引言、正文、结论
- 语言流畅，通俗易懂
- 突出关键信息

文章:"""
        
        response = self.llm.invoke(prompt)
        return response.content


class ReviewerAgent:
    """审核 Agent：负责质量检查"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.3
        )
    
    def run(self, content: str) -> str:
        """执行审核任务"""
        print(f"🔍 [ReviewerAgent] 审核内容")
        
        prompt = f"""作为审核员，请对以下内容进行质量评估：

内容:
{content[:1000]}...

请评估：
1. 内容准确性
2. 结构完整性
3. 语言流畅度
4. 改进建议

审核意见:"""
        
        response = self.llm.invoke(prompt)
        return response.content


# 初始化 Agents
research_agent = ResearchAgent()
writer_agent = WriterAgent()
reviewer_agent = ReviewerAgent()


# ========== 3. 定义节点 ==========
def classify_task(state: MultiAgentState) -> MultiAgentState:
    """任务分类节点"""
    print("📋 [Classify] 分析任务类型...")
    
    # 获取用户输入
    user_input = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    # 简单分类
    if "研究" in user_input or "分析" in user_input:
        task_type = "research"
    elif "写" in user_input or "文章" in user_input:
        task_type = "writing"
    else:
        task_type = "full"  # 完整流程
    
    print(f"   任务类型: {task_type}")
    
    return {
        **state,
        "task_type": task_type
    }


def run_research(state: MultiAgentState) -> MultiAgentState:
    """运行研究 Agent"""
    print("\n" + "="*50)
    print("阶段 1: 研究")
    print("="*50)
    
    # 提取主题
    topic = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            topic = msg.content
            break
    
    result = research_agent.run(topic)
    
    return {
        **state,
        "research_result": result,
        "current_agent": "research",
        "messages": [AIMessage(content=f"研究完成: {result[:100]}...")]
    }


def run_writer(state: MultiAgentState) -> MultiAgentState:
    """运行写作 Agent"""
    print("\n" + "="*50)
    print("阶段 2: 写作")
    print("="*50)
    
    # 提取主题
    topic = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            topic = msg.content
            break
    
    research = state.get("research_result", "")
    result = writer_agent.run(topic, research)
    
    return {
        **state,
        "writer_result": result,
        "current_agent": "writer",
        "messages": [AIMessage(content=f"文章完成: {result[:100]}...")]
    }


def run_reviewer(state: MultiAgentState) -> MultiAgentState:
    """运行审核 Agent"""
    print("\n" + "="*50)
    print("阶段 3: 审核")
    print("="*50)
    
    content = state.get("writer_result", "")
    result = reviewer_agent.run(content)
    
    return {
        **state,
        "reviewer_result": result,
        "current_agent": "reviewer",
        "messages": [AIMessage(content=f"审核完成: {result[:100]}...")]
    }


def finalize(state: MultiAgentState) -> MultiAgentState:
    """整合最终结果"""
    print("\n" + "="*50)
    print("阶段 4: 整合")
    print("="*50)
    
    final = f"""# 最终成果

## 研究摘要
{state.get('research_result', '')[:200]}...

## 正文内容
{state.get('writer_result', '')[:300]}...

## 审核意见
{state.get('reviewer_result', '')[:200]}...
"""
    
    return {
        **state,
        "final_output": final,
        "messages": [AIMessage(content="多 Agent 协作完成！")]
    }


# ========== 4. 路由函数 ==========
def route_by_task(state: MultiAgentState) -> Literal["research", "writer", "finalize"]:
    """根据任务类型路由"""
    task_type = state.get("task_type", "full")
    
    if task_type == "research":
        return "research"
    elif task_type == "writing":
        return "writer"
    else:
        return "research"  # 完整流程从研究开始


def route_after_research(state: MultiAgentState) -> str:
    """研究后路由"""
    task_type = state.get("task_type", "full")
    if task_type == "research":
        return "finalize"
    return "writer"


def route_after_writer(state: MultiAgentState) -> str:
    """写作后路由"""
    task_type = state.get("task_type", "full")
    if task_type == "writing":
        return "finalize"
    return "reviewer"


# ========== 5. 构建多 Agent 图 ==========
def build_multi_agent_graph():
    """构建多 Agent 协作图"""
    
    workflow = StateGraph(MultiAgentState)
    
    # 添加节点
    workflow.add_node("classify", classify_task)
    workflow.add_node("research", run_research)
    workflow.add_node("writer", run_writer)
    workflow.add_node("reviewer", run_reviewer)
    workflow.add_node("finalize", finalize)
    
    # 设置入口
    workflow.set_entry_point("classify")
    
    # 分类后路由
    workflow.add_conditional_edges(
        "classify",
        route_by_task,
        {
            "research": "research",
            "writer": "writer",
            "full": "research"
        }
    )
    
    # 研究后路由
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "finalize": "finalize",
            "writer": "writer"
        }
    )
    
    # 写作后路由
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        {
            "finalize": "finalize",
            "reviewer": "reviewer"
        }
    )
    
    # 审核后整合
    workflow.add_edge("reviewer", "finalize")
    workflow.add_edge("finalize", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ========== 6. 运行演示 ==========
def main():
    print("=" * 60)
    print("LangGraph 核心概念 06 - 多 Agent 系统")
    print("=" * 60)
    
    graph = build_multi_agent_graph()
    
    # 测试完整流程
    print("\n📝 测试: 完整协作流程\n")
    
    config = {"configurable": {"thread_id": "multi_agent_demo"}}
    
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="写一篇关于人工智能的文章")],
            "current_agent": "",
            "task_type": "",
            "research_result": "",
            "writer_result": "",
            "reviewer_result": "",
            "final_output": ""
        },
        config=config
    )
    
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    print(state["final_output"][:500] + "...")
    
    # 展示架构
    print("\n" + "="*60)
    print("多 Agent 系统架构")
    print("="*60)
    print("""
    ┌─────────────┐
    │   任务分类   │
    └──────┬──────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│  研究员  │  │  写手   │
│ (Research)│  │ (Writer)│
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
    ┌─────────────┐
    │   审核员    │
    │ (Reviewer)  │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   整合输出   │
    └─────────────┘
    """)


if __name__ == "__main__":
    main()
    print("\n✅ 多 Agent 系统测试完成！")
