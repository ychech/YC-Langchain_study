#!/usr/bin/env python3
"""
LangChain 1.0.x 快速上手 - 带记忆的对话
功能：实现多轮对话，AI记住上下文

注意：LangChain 1.0 推荐使用新的记忆管理方式，通过 trim_messages 控制上下文长度
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta/v1",
    temperature=0.7,
    max_tokens=2048
)

# 构建带记忆占位符的提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的助手，记得之前的对话内容。"),
    MessagesPlaceholder(variable_name="messages"),  # 存放对话历史
])

# 创建消息修剪器（控制上下文长度，避免超出 token 限制）
# 注意：DeepSeek 不支持 token 计数，使用简单的消息数量限制
MAX_HISTORY = 10  # 最多保留 10 条消息

#方案二
"""
用token数量
"""

#方案三 当历史太长时，用 AI 总结早期对话
"""
if len(history) > 20:
    early_history = history[:10]
    summary = llm.invoke(f"总结这些对话：{early_history}")
    history = [SystemMessage(content=f"之前对话摘要：{summary}")] + history[10:]
"""



# 消息修剪器
def trim_history(messages):
    """简单的消息修剪，保留最近的消息"""
    if len(messages) > MAX_HISTORY:
        # 保留 system 消息（如果有的话）和最近的消息
        # 负数索引切片，保留后10 条消息
        return messages[-MAX_HISTORY:]
    return messages

#RunnablePassthrough 是 LangChain 中Runnable接口的一个实现，它的核心作用是：
#透传数据：把输入的数据原封不动地传递到下一个环节
#增强数据：在透传的同时，新增或修改部分字段（结合assign方法，assign增强数据，增加字段）
#因为 LCEL 的链式调用（|操作符）要求每一步都是Runnable对象，字典不行
#lambda是函数的简写，匿名函数，lambda x: x是定义一个函数，参数是x，返回x

# 构建链：修剪消息 -> 提示词 -> LLM
chain = RunnablePassthrough.assign(messages=lambda x: trim_history(x["messages"])) | prompt | llm

# 模拟对话历史存储（实际应用中可用数据库）
conversation_store = {}

# 对话历史处理函数
def get_or_create_history(session_id: str):
    """获取或创建对话历史"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]

# 聊天处理函数
def chat(session_id: str, user_input: str):
    """进行一次对话"""
    # 获取历史
    history = get_or_create_history(session_id)
    

    # 还支持其他类型聊天分类
    #     SystemMessage,      # 系统指令（设定 AI 角色）
    #     HumanMessage,       # 用户消息
    #     AIMessage,          # AI 消息
    #     ToolMessage,        # 工具调用结果
    #     FunctionMessage,    # 函数调用结果
    from langchain_core.messages import HumanMessage, AIMessage

    # 添加用户消息
    history.append(HumanMessage(content=user_input))

    print(f"历史：{history}")
    # 调用链
    response = chain.invoke({"messages": history})
    
    # 添加 AI 回复到历史
    history.append(AIMessage(content=response.content))
    
    return response.content

# ========== 测试多轮对话 ==========
print("===== 多轮对话测试 =====\n")

session_id = "user_001"

# 第一轮
print("用户：我叫张三，请记住我的名字")
print(f"AI：{chat(session_id, '我叫张三，请记住我的名字')}\n")

# 第二轮（验证记忆）
print("用户：我叫什么名字？")
print(f"AI：{chat(session_id, '我叫什么名字？')}\n")

# 第三轮（继续对话）
print("用户：我喜欢Python编程")
print(f"AI：{chat(session_id, '我喜欢Python编程')}\n")

# 验证记忆是否持续
print("用户：总结一下你记得关于我的信息")
print(f"AI：{chat(session_id, '总结一下你记得关于我的信息')}\n")

print("✅ 记忆对话测试完成")
print(f"📊 当前会话历史消息数：{len(conversation_store[session_id])}")
