#!/usr/bin/env python3
"""
LangGraph 核心概念 08 - 流式事件处理
功能：实时获取图的执行事件，实现渐进式输出

核心概念：
- stream: 流式执行，实时获取状态变化
- events: 事件类型（on_chain_start, on_chain_end 等）
- 实时反馈: 用户可以看到每一步的执行
"""
from typing import TypedDict, Annotated, List, AsyncIterator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


# ========== 1. 定义状态 ==========
class StreamState(TypedDict):
    """流式处理状态"""
    input: str
    output: str
    steps: List[str]  # 执行步骤记录


# ========== 2. 定义节点 ==========
def step1_node(state: StreamState) -> StreamState:
    """步骤 1"""
    print("⚡ [Step 1] 执行第一步...")
    return {
        **state,
        "steps": ["步骤 1 完成"]
    }


def step2_node(state: StreamState) -> StreamState:
    """步骤 2"""
    print("⚡ [Step 2] 执行第二步...")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    response = llm.invoke(f"请用一句话回答: {state['input']}")
    
    return {
        **state,
        "output": response.content,
        "steps": state["steps"] + ["步骤 2 完成"]
    }


def step3_node(state: StreamState) -> StreamState:
    """步骤 3"""
    print("⚡ [Step 3] 执行第三步...")
    return {
        **state,
        "steps": state["steps"] + ["步骤 3 完成"]
    }


# ========== 3. 构建图 ==========
def build_stream_graph():
    """构建用于流式演示的图"""
    
    workflow = StateGraph(StreamState)
    
    workflow.add_node("step1", step1_node)
    workflow.add_node("step2", step2_node)
    workflow.add_node("step3", step3_node)
    
    workflow.set_entry_point("step1")
    workflow.add_edge("step1", "step2")
    workflow.add_edge("step2", "step3")
    workflow.add_edge("step3", END)
    
    return workflow.compile()


# ========== 4. 流式执行演示 ==========
def demo_stream():
    """演示流式执行"""
    print("=" * 60)
    print("LangGraph 核心概念 08 - 流式事件处理")
    print("=" * 60)
    
    graph = build_stream_graph()
    
    print("\n📊 方式 1: 使用 stream() 查看状态变化\n")
    print("-" * 50)
    
    initial_state = {
        "input": "什么是人工智能？",
        "output": "",
        "steps": []
    }
    
    # 流式执行
    for event in graph.stream(initial_state):
        print(f"\n📦 事件: {event}")
        print(f"   当前步骤: {event.get('steps', [])}")
        if event.get('output'):
            print(f"   当前输出: {event['output'][:50]}...")


# ========== 5. 事件类型演示 ==========
def demo_event_types():
    """演示不同类型的事件"""
    print("\n" + "=" * 60)
    print("事件类型说明")
    print("=" * 60)
    
    print("""
LangGraph 支持以下事件类型:

1. on_chain_start / on_chain_end
   - 链/图开始和结束

2. on_chain_stream
   - 链的中间状态更新

3. on_llm_start / on_llm_end
   - LLM 调用开始和结束

4. on_llm_stream
   - LLM 的 token 级流式输出

5. on_tool_start / on_tool_end
   - 工具调用开始和结束

6. on_prompt_start / on_prompt_end
   - 提示词构建开始和结束
    """)


# ========== 6. 异步流式演示 ==========
async def demo_async_stream():
    """异步流式执行演示"""
    print("\n" + "=" * 60)
    print("异步流式执行")
    print("=" * 60)
    
    graph = build_stream_graph()
    
    initial_state = {
        "input": "Python 有什么优点？",
        "output": "",
        "steps": []
    }
    
    print("\n⚡ 异步流式执行:\n")
    
    # 异步流式执行
    async for event in graph.astream(initial_state):
        print(f"📦 事件: {list(event.keys())}")
        for key, value in event.items():
            if isinstance(value, dict):
                if value.get('steps'):
                    print(f"   步骤: {value['steps']}")
                if value.get('output'):
                    print(f"   输出: {value['output'][:50]}...")


# ========== 7. 实际应用场景 ==========
def show_use_cases():
    """展示流式处理的实际应用场景"""
    print("\n" + "=" * 60)
    print("流式处理的实际应用场景")
    print("=" * 60)
    
    use_cases = [
        {
            "场景": "聊天机器人",
            "描述": "实时显示思考过程，让用户知道 AI 正在工作",
            "代码示例": """
for event in graph.stream({"message": user_input}):
    if "thinking" in event:
        yield f"思考中: {event['thinking']}"
    if "response" in event:
        yield f"回答: {event['response']}"
            """
        },
        {
            "场景": "进度显示",
            "描述": "长任务执行时显示进度",
            "代码示例": """
for event in graph.stream(task):
    progress = event.get("progress", 0)
    update_progress_bar(progress)
            """
        },
        {
            "场景": "调试监控",
            "描述": "开发时监控每个节点的执行",
            "代码示例": """
for event in graph.stream(input):
    print(f"节点 {event['__name__']} 完成")
    print(f"状态: {event['state']}")
            """
        },
        {
            "场景": "人机协作",
            "描述": "在关键节点暂停等待人工确认",
            "代码示例": """
for event in graph.stream(input):
    if event.get("needs_human"):
        human_input = await get_human_input()
        graph.invoke(Command(resume=human_input))
            """
        }
    ]
    
    for i, case in enumerate(use_cases, 1):
        print(f"\n{i}. {case['场景']}")
        print(f"   描述: {case['描述']}")
        print(f"   代码示例: {case['代码示例']}")


# ========== 8. Web 应用集成示例 ==========
def show_web_integration():
    """展示 Web 应用集成示例"""
    print("\n" + "=" * 60)
    print("Web 应用集成示例 (FastAPI + SSE)")
    print("=" * 60)
    
    print("""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    async def event_generator():
        graph = build_chat_graph()
        
        async for event in graph.astream({"input": message}):
            # 将事件转换为 SSE 格式
            yield f"data: {json.dumps(event)}\\n\\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# 前端 JavaScript:
# const eventSource = new EventSource('/chat?message=你好');
# eventSource.onmessage = (event) => {
#     const data = JSON.parse(event.data);
#     updateUI(data);
# };
    """)


def main():
    demo_stream()
    demo_event_types()
    
    # 运行异步演示
    asyncio.run(demo_async_stream())
    
    show_use_cases()
    show_web_integration()
    
    print("\n✅ 流式事件处理演示完成！")


if __name__ == "__main__":
    main()
