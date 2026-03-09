#!/usr/bin/env python3
"""
LangChain 1.0.x 快速上手 - Deep Agent (深度代理)
功能：使用新的 create_agent API 创建智能代理

官方文档格式：
    agent.invoke({"messages": [{"role": "user", "content": "..."}]})
"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import math

load_dotenv()

# ========== 定义工具 ==========

@tool
# 数学计算
def calculator(expression: str) -> str:
    """
    执行数学计算，支持加减乘除、幂运算、开方等。
    
    Args:
        expression: 数学表达式，如 "123 * 456" 或 "sqrt(16)"
    """
    try:
        expression = expression.strip()
        allowed_names = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "pi": math.pi,
            "e": math.e
        }
        # eval是执行表达式，但是不安全
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
# 天气查询
def get_weather(city: str) -> str:
    """
    查询指定城市的天气情况。
    
    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    weather_db = {
        "北京": "晴天，25°C，空气质量良",
        "上海": "多云，28°C，可能有阵雨",
        "广州": "雷阵雨，30°C，湿度较高",
        "深圳": "阴天，29°C，东风3级",
    }
    return weather_db.get(city, f"暂无 {city} 的天气数据")

@tool
# 信息搜索
def search_info(query: str) -> str:
    """
    搜索信息。
    
    Args:
        query: 搜索关键词
    """
    knowledge_base = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "langchain": "LangChain 是一个用于开发 LLM 应用的框架，1.0 版本于 2025年10月发布。",
        "deepseek": "DeepSeek 是一家中国 AI 公司，开发了 DeepSeek-V3 等大语言模型。",
    }
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return f"关于 '{query}' 的搜索结果：建议查阅最新资料。"

# 工具列表
tools = [calculator, get_weather, search_info]

# ========== 初始化模型 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1,
    max_tokens=2048
)

# ========== 创建 Agent ==========
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一位智能助手，可以使用以下工具帮助用户：

1. calculator: 数学计算工具 - 用于执行数学计算
2. get_weather: 天气查询工具 - 用于查询城市天气
3. search_info: 信息搜索工具 - 用于搜索知识信息

请根据用户需求选择合适的工具，并在获取结果后用自然语言回答用户。
"""
)

# ========== 测试 Agent ==========
print("===== Deep Agent 测试（LangChain 1.0）=====\n")

def run_agent(question: str) -> str:
    """
    运行 agent 并获取最终回答
    
    官方格式: agent.invoke({"messages": [{"role": "user", "content": question}]})
    """
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # 获取最后一条 AI 消息
    messages = response["messages"]
    for msg in reversed(messages):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "无响应"

test_questions = [
    "计算 123 乘以 456 等于多少？",
    "北京今天的天气怎么样？",
    "LangChain 是什么？",
    "如果我有 1000 元，买了 3 件每件 199 元的衣服，还剩多少钱？",
]

for question in test_questions:
    print(f"\n用户：{question}")
    answer = run_agent(question)
    print(f"AI：{answer}")
    print("-" * 60)

# ========== 带会话记忆的 Agent ==========
print("\n===== 带记忆的 Agent 测试 =====\n")

conversation_history = {}

def chat_with_memory(session_id: str, user_input: str):
    """带记忆的对话"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # 构建消息历史
    messages = []
    for msg in conversation_history[session_id][-4:]:  # 最近2轮
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["ai"]})
    
    # 添加当前问题
    messages.append({"role": "user", "content": user_input})
    
    # 调用 agent
    response = agent.invoke({"messages": messages})
    
    # 获取 AI 回复
    ai_content = ""
    for msg in reversed(response["messages"]):
        if msg.type == "ai" and msg.content:
            ai_content = msg.content
            break
    
    # 保存对话
    conversation_history[session_id].append({
        "user": user_input,
        "ai": ai_content
    })
    
    return ai_content

# 测试记忆
session = "test_session"
print(f"用户：我叫李四")
print(f"AI：{chat_with_memory(session, '我叫李四')}\n")

print(f"用户：我叫什么名字？")
print(f"AI：{chat_with_memory(session, '我叫什么名字？')}\n")

print("✅ Agent 测试完成")
