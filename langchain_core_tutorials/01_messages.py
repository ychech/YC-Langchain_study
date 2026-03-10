#!/usr/bin/env python3
"""
LangChain Core 核心概念 01 - 消息类型 (Messages)
功能：理解不同类型的消息及其作用

核心概念：
- SystemMessage: 系统消息，设定 AI 角色和行为
- HumanMessage: 人类消息，用户输入
- AIMessage: AI 消息，模型回复
- ToolMessage: 工具消息，工具调用结果

学习要点：
1. 理解不同消息类型的用途和区别
2. 掌握消息的创建和使用方法
3. 学会消息的序列化和反序列化
4. 了解消息的属性和元数据
"""

# =============================================================================
# 模块导入
# =============================================================================

# 从 langchain_core.messages 导入核心消息类型
# SystemMessage: 系统级消息，用于设定 AI 的行为和角色
# HumanMessage: 人类用户发送的消息
# AIMessage: AI 助手回复的消息
# ToolMessage: 工具调用返回的结果消息
# messages_to_dict/messages_from_dict: 消息的序列化和反序列化工具
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_to_dict,
    messages_from_dict
)

# ChatOpenAI: LangChain 封装的 OpenAI 兼容聊天模型接口
from langchain_openai import ChatOpenAI

# 环境变量和路径处理
from dotenv import load_dotenv
import os
from pathlib import Path

# =============================================================================
# 环境配置
# =============================================================================

# 加载环境变量
# 首先尝试从当前文件的父目录的父目录（项目根目录）加载 .env 文件
# 如果不存在，则尝试从系统环境变量加载
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


# =============================================================================
# 函数定义：演示不同消息类型的创建
# =============================================================================

def demo_message_types():
    """
    演示不同消息类型的创建和基本属性
    
    本函数展示 LangChain 中四种核心消息类型的创建方式：
    1. SystemMessage - 系统消息：定义 AI 的角色和行为准则
    2. HumanMessage - 人类消息：表示用户的输入
    3. AIMessage - AI 消息：表示 AI 的回复
    4. ToolMessage - 工具消息：表示工具调用的结果
    
    消息类型是构建对话的基础，每种类型在对话中扮演不同角色。
    """
    print("=" * 60)
    print("LangChain Core 核心概念 01 - 消息类型")
    print("=" * 60)
    
    # ========== 1. 创建不同类型的消息 ==========
    # 消息是 LangChain 对话系统的基本单元
    # 不同类型的消息用于区分对话中不同角色的发言
    print("\n📨 创建不同类型的消息\n")
    print("-" * 50)
    
    # -------------------------------------------------------------------------
    # SystemMessage: 系统消息
    # 用途：设定 AI 的系统提示词（System Prompt）
    # 特点：
    #   - 通常放在消息列表的最开始
    #   - 定义 AI 的角色、性格、专业知识领域
    #   - 可以设定回答格式、约束条件等
    # 示例场景：
    #   - "你是一位专业的 Python 程序员"
    #   - "请用简洁的语言回答问题"
    #   - "你只能回答技术相关的问题"
    # -------------------------------------------------------------------------
    system_msg = SystemMessage(content="你是一位专业的 Python 编程助手，回答简洁明了。")
    print(f"\n1️⃣ SystemMessage (系统消息):")
    print(f"   类型: {type(system_msg).__name__}")
    print(f"   内容: {system_msg.content}")
    print(f"   作用: 设定 AI 的角色和行为")
    
    # -------------------------------------------------------------------------
    # HumanMessage: 人类消息
    # 用途：表示用户的输入/问题
    # 特点：
    #   - 代表真实用户的提问或指令
    #   - 在对话历史中，HumanMessage 和 AIMessage 通常交替出现
    #   - 是触发 AI 回复的主要输入
    # 示例场景：
    #   - "什么是装饰器？"
    #   - "请帮我写一个排序函数"
    #   - "解释这段代码"
    # -------------------------------------------------------------------------
    human_msg = HumanMessage(content="什么是装饰器？")
    print(f"\n2️⃣ HumanMessage (人类消息):")
    print(f"   类型: {type(human_msg).__name__}")
    print(f"   内容: {human_msg.content}")
    print(f"   作用: 用户的输入/问题")
    
    # -------------------------------------------------------------------------
    # AIMessage: AI 消息
    # 用途：表示 AI 助手的回复
    # 特点：
    #   - 包含模型生成的回复内容
    #   - 可以包含额外的元数据（如 token 使用量、模型名称等）
    #   - 在对话历史中记录 AI 的回复，用于保持上下文
    # 示例场景：
    #   - AI 对问题的回答
    #   - AI 生成的代码
    #   - AI 的确认或澄清
    # -------------------------------------------------------------------------
    ai_msg = AIMessage(content="装饰器是 Python 中用于修改函数或类行为的语法糖。")
    print(f"\n3️⃣ AIMessage (AI 消息):")
    print(f"   类型: {type(ai_msg).__name__}")
    print(f"   内容: {ai_msg.content}")
    print(f"   作用: AI 的回复/输出")
    
    # -------------------------------------------------------------------------
    # ToolMessage: 工具消息
    # 用途：表示工具调用的返回结果
    # 特点：
    #   - 当 AI 调用外部工具（如计算器、搜索引擎、数据库等）时使用
    #   - 必须包含 tool_call_id，用于关联对应的工具调用请求
    #   - 可以包含工具名称（name）便于识别
    # 使用场景：
    #   - AI 调用计算器后的计算结果
    #   - AI 调用搜索引擎后的搜索结果
    #   - AI 调用 API 后的返回数据
    # 注意：ToolMessage 通常紧跟在包含 tool_calls 的 AIMessage 之后
    # -------------------------------------------------------------------------
    tool_msg = ToolMessage(
        content="计算结果: 42",      # 工具返回的实际内容
        tool_call_id="call_123",    # 工具调用的唯一标识，用于关联请求和响应
        name="calculator"           # 工具名称，便于识别是哪个工具返回的结果
    )
    print(f"\n4️⃣ ToolMessage (工具消息):")
    print(f"   类型: {type(tool_msg).__name__}")
    print(f"   内容: {tool_msg.content}")
    print(f"   工具ID: {tool_msg.tool_call_id}")
    print(f"   工具名: {tool_msg.name}")
    print(f"   作用: 工具调用的返回结果")


# =============================================================================
# 函数定义：演示消息的实际使用
# =============================================================================

def demo_message_usage():
    """
    演示消息在实际对话中的使用
    
    本函数展示如何：
    1. 构建完整的对话历史消息列表
    2. 使用消息列表调用 LLM 模型
    3. 处理模型的回复
    
    关键概念：
    - 消息列表（Message List）：按时间顺序排列的消息序列
    - 对话上下文：模型通过消息列表理解对话历史
    - 角色交替：HumanMessage 和 AIMessage 交替出现
    """
    print("\n" + "=" * 60)
    print("消息的实际使用")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 构建消息列表
    # 消息列表是对话的核心数据结构，包含完整的对话历史
    # 顺序很重要：消息按时间顺序排列，系统消息通常在开头
    # 
    # 典型结构：
    # [SystemMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
    # 
    # 注意事项：
    # 1. SystemMessage 通常只有一个，放在最前面
    # 2. HumanMessage 和 AIMessage 成对出现
    # 3. 最后一个消息通常是 HumanMessage（用户最新的问题）
    # -------------------------------------------------------------------------
    messages = [
        # 系统消息：定义 AI 的角色和行为
        SystemMessage(content="你是一位友好的助手。"),
        
        # 第一轮对话：用户问候
        HumanMessage(content="你好！"),
        
        # 第一轮对话：AI 回复
        AIMessage(content="你好！很高兴见到你。有什么可以帮助你的吗？"),
        
        # 第二轮对话：用户新问题
        HumanMessage(content="今天天气怎么样？")
    ]
    
    # 打印对话历史，帮助理解消息列表的结构
    print("\n📜 消息对话历史:\n")
    for i, msg in enumerate(messages, 1):
        # 根据消息类型显示不同的角色图标
        role = "🤖 系统" if isinstance(msg, SystemMessage) else \
               "👤 用户" if isinstance(msg, HumanMessage) else \
               "🤖 AI" if isinstance(msg, AIMessage) else "🔧 工具"
        # 只显示内容的前 50 个字符，避免输出过长
        print(f"{i}. {role}: {msg.content[:50]}...")
    
    # -------------------------------------------------------------------------
    # 使用模型进行对话
    # ChatOpenAI 是 LangChain 封装的聊天模型接口
    # 支持 OpenAI API 以及兼容 OpenAI API 格式的第三方服务（如 DeepSeek）
    # 
    # 参数说明：
    # - model: 模型名称，如 "gpt-4", "deepseek-chat" 等
    # - api_key: API 密钥，从环境变量读取
    # - base_url: API 的基础 URL，用于第三方服务
    # -------------------------------------------------------------------------
    print("\n🚀 使用消息列表调用模型:\n")
    
    # 创建 LLM 实例
    # 这里使用 DeepSeek API 作为示例，也可以使用 OpenAI 或其他兼容服务
    llm = ChatOpenAI(
        model="deepseek-chat",                      # 模型名称
        api_key=os.getenv("DEEPSEEK_API_KEY"),      # 从环境变量获取 API 密钥
        base_url="https://api.deepseek.com/v1"      # DeepSeek API 地址
    )
    
    # 调用模型
    # invoke 方法接收消息列表，返回 AIMessage 对象
    # 模型会根据完整的对话历史生成回复
    response = llm.invoke(messages)
    print(f"🤖 AI 回复: {response.content}")


# =============================================================================
# 函数定义：演示消息的序列化和反序列化
# =============================================================================

def demo_message_serialization():
    """
    演示消息的序列化和反序列化
    
    本函数展示如何：
    1. 将消息列表转换为字典列表（便于 JSON 序列化）
    2. 从字典列表恢复消息对象
    
    应用场景：
    - 保存对话历史到数据库或文件
    - 通过网络传输对话数据
    - 缓存对话状态
    
    关键函数：
    - messages_to_dict(): 将消息对象列表转换为字典列表
    - messages_from_dict(): 从字典列表恢复消息对象
    """
    print("\n" + "=" * 60)
    print("消息序列化 (用于保存和传输)")
    print("=" * 60)
    
    # 创建示例消息列表
    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好"),
        AIMessage(content="你好！")
    ]
    
    # -------------------------------------------------------------------------
    # 序列化：将消息对象转换为字典
    # 
    # messages_to_dict 函数将每个消息对象转换为字典格式：
    # {
    #     "type": "system" | "human" | "ai" | "tool",
    #     "data": {
    #         "content": "消息内容",
    #         "additional_kwargs": {...},
    #         ...
    #     }
    # }
    # 
    # 字典格式可以方便地：
    # - 使用 json.dumps() 转换为 JSON 字符串
    # - 存储到数据库（如 MongoDB、Redis）
    # - 通过 HTTP API 传输
    # -------------------------------------------------------------------------
    print("\n📦 转换为字典:\n")
    dict_list = messages_to_dict(messages)
    for msg_dict in dict_list:
        print(f"   {msg_dict}")
    
    # -------------------------------------------------------------------------
    # 反序列化：从字典恢复消息对象
    # 
    # messages_from_dict 函数根据字典中的 type 字段：
    # - 自动识别消息类型
    # - 创建对应的消息对象
    # - 保留所有原始数据
    # 
    # 这样可以从数据库或文件中恢复完整的对话历史
    # -------------------------------------------------------------------------
    print("\n📥 从字典恢复:\n")
    restored_messages = messages_from_dict(dict_list)
    for msg in restored_messages:
        print(f"   {type(msg).__name__}: {msg.content}")


# =============================================================================
# 函数定义：演示消息的属性和方法
# =============================================================================

def demo_message_properties():
    """
    演示消息的属性和方法
    
    本函数展示 AIMessage 的常用属性：
    - content: 消息的主要内容
    - type: 消息类型标识
    - additional_kwargs: 额外的关键字参数
    - response_metadata: 响应元数据（如模型信息、token 使用量等）
    - id: 消息的唯一标识符
    
    这些属性对于调试、日志记录和高级功能非常重要。
    """
    print("\n" + "=" * 60)
    print("消息属性和方法")
    print("=" * 60)
    
    # 创建一个包含完整属性的 AIMessage 示例
    # 注意：实际从模型获取的回复会自动填充这些属性
    msg = AIMessage(
        # content: 消息的主要内容，即 AI 的回复文本
        content="这是一个 AI 回复",
        
        # additional_kwargs: 额外的关键字参数
        # 可用于存储自定义数据，如置信度、来源等
        additional_kwargs={"confidence": 0.95},
        
        # response_metadata: 响应元数据
        # 通常包含模型信息、token 使用量、完成原因等
        response_metadata={"model": "gpt-4"}
    )
    
    # 打印消息的各项属性
    print(f"\n📋 消息属性:\n")
    
    # content: 消息的核心内容
    print(f"   content: {msg.content}")
    
    # type: 消息类型，如 "ai", "human", "system", "tool"
    print(f"   type: {msg.type}")
    
    # additional_kwargs: 额外的关键字参数字典
    print(f"   additional_kwargs: {msg.additional_kwargs}")
    
    # response_metadata: 响应元数据字典
    print(f"   response_metadata: {msg.response_metadata}")
    
    # id: 消息的唯一标识符，自动生成的 UUID
    print(f"   id: {msg.id}")


# =============================================================================
# 主函数：程序入口
# =============================================================================

def main():
    """
    主函数：演示消息类型的各种用法
    
    按顺序执行以下演示：
    1. demo_message_types: 展示四种消息类型的创建
    2. demo_message_usage: 展示消息在实际对话中的使用
    3. demo_message_serialization: 展示消息的序列化和反序列化
    4. demo_message_properties: 展示消息的属性和元数据
    
    最后输出总结表格和最佳实践提示。
    """
    # 演示 1: 消息类型的创建
    demo_message_types()
    
    # 演示 2: 消息的实际使用
    demo_message_usage()
    
    # 演示 3: 消息序列化
    demo_message_serialization()
    
    # 演示 4: 消息属性
    demo_message_properties()
    
    # =============================================================================
    # 总结和最佳实践
    # =============================================================================
    print("\n" + "=" * 60)
    print("消息类型总结")
    print("=" * 60)
    print("""
┌─────────────────┬─────────────────────────────────────────┐
│ 消息类型         │ 用途                                    │
├─────────────────┼─────────────────────────────────────────┤
│ SystemMessage   │ 设定 AI 角色、行为、约束                 │
│ HumanMessage    │ 用户输入/问题                           │
│ AIMessage       │ AI 回复/输出                            │
│ ToolMessage     │ 工具调用结果                            │
└─────────────────┴─────────────────────────────────────────┘

💡 最佳实践:
1. SystemMessage 放在消息列表最前面
   - 系统消息定义了 AI 的整体行为
   - 放在开头确保模型首先"看到"角色设定

2. HumanMessage 和 AIMessage 交替出现
   - 保持对话的连贯性
   - 帮助模型理解对话流程

3. ToolMessage 紧跟在 AIMessage 之后
   - ToolMessage 是对工具调用的响应
   - 需要与对应的 tool_call_id 关联

4. 注意消息列表长度
   - 过长的对话历史会消耗更多 token
   - 考虑使用对话压缩或摘要技术

5. 序列化时保存完整元数据
   - 使用 messages_to_dict 保留所有信息
   - 便于后续分析和调试
    """)
    
    print("\n✅ 消息类型学习完成！")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    # 当脚本直接运行时执行 main 函数
    # 当作为模块导入时，main 函数不会自动执行
    main()
