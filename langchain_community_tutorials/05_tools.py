#!/usr/bin/env python3
"""
LangChain Community 核心概念 05 - 工具集成 (Tools)
功能：学习使用各种内置工具和自定义工具

核心概念：
- @tool 装饰器: 创建自定义工具
- BaseTool: 工具基类
- 工具调用: ToolMessage 和函数调用
"""
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import math
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


# ========== 使用 @tool 装饰器创建工具 ==========
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算，支持加减乘除、幂运算、开方等。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"
    """
    try:
        # 安全计算环境
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
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气情况（模拟）。
    
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
def search_web(query: str) -> str:
    """
    使用 DuckDuckGo 搜索网络信息。
    
    Args:
        query: 搜索关键词
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索失败: {str(e)}"


# ========== 使用 BaseTool 创建工具 ==========
class TranslateInput(BaseModel):
    """翻译工具输入参数"""
    text: str = Field(description="要翻译的文本")
    target_language: str = Field(description="目标语言，如 'en'、'zh'、'ja'")


class TranslateTool(BaseTool):
    """翻译工具"""
    name: str = "translator"
    description: str = "翻译文本到指定语言"
    args_schema: Type[BaseModel] = TranslateInput
    
    def _run(self, text: str, target_language: str) -> str:
        """执行翻译"""
        # 模拟翻译
        translations = {
            ("hello", "zh"): "你好",
            ("world", "zh"): "世界",
            ("你好", "en"): "hello",
            ("谢谢", "en"): "thank you",
        }
        key = (text.lower(), target_language)
        result = translations.get(key, f"[模拟翻译] {text} -> {target_language}")
        return result


def demo_tool_decorator():
    """演示 @tool 装饰器"""
    print("=" * 60)
    print("LangChain Community 核心概念 05 - 工具集成")
    print("=" * 60)
    
    print("\n🔧 使用 @tool 装饰器创建工具\n")
    print("-" * 50)
    
    print("\n1️⃣ 查看工具信息:")
    print(f"   工具名称: {calculator.name}")
    print(f"   工具描述: {calculator.description}")
    print(f"   参数模式: {calculator.args}")
    
    print("\n2️⃣ 调用工具:")
    result = calculator.invoke({"expression": "2 + 3 * 4"})
    print(f"   输入: 2 + 3 * 4")
    print(f"   输出: {result}")
    
    print("\n3️⃣ 另一个工具:")
    result = get_weather.invoke({"city": "北京"})
    print(f"   输入: 北京")
    print(f"   输出: {result}")


def demo_base_tool():
    """演示 BaseTool"""
    print("\n" + "=" * 60)
    print("使用 BaseTool 创建复杂工具")
    print("=" * 60)
    
    print("\n🏗️ 继承 BaseTool 创建工具:\n")
    
    translator = TranslateTool()
    
    print("   工具名称:", translator.name)
    print("   工具描述:", translator.description)
    print("   参数模式:", translator.args_schema.schema())
    
    print("\n   调用翻译工具:")
    result = translator.invoke({"text": "hello", "target_language": "zh"})
    print(f"   输入: text='hello', target_language='zh'")
    print(f"   输出: {result}")


def demo_tool_list():
    """演示工具列表"""
    print("\n" + "=" * 60)
    print("工具列表管理")
    print("=" * 60)
    
    print("\n📋 创建工具列表:\n")
    
    tools = [calculator, get_weather, TranslateTool()]
    
    print(f"   工具数量: {len(tools)}\n")
    
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool.name}")
        print(f"      描述: {tool.description[:50]}...")


def demo_community_tools():
    """演示社区工具"""
    print("\n" + "=" * 60)
    print("LangChain Community 内置工具")
    print("=" * 60)
    
    print("\n🛠️ 常用内置工具:\n")
    
    tools_info = [
        {
            "名称": "DuckDuckGoSearchRun",
            "用途": "网络搜索",
            "导入": "from langchain_community.tools import DuckDuckGoSearchRun"
        },
        {
            "名称": "WikipediaQueryRun",
            "用途": "维基百科查询",
            "导入": "from langchain_community.tools import WikipediaQueryRun"
        },
        {
            "名称": "ShellTool",
            "用途": "执行 shell 命令",
            "导入": "from langchain_community.tools import ShellTool"
        },
        {
            "名称": "PythonREPLTool",
            "用途": "执行 Python 代码",
            "导入": "from langchain_community.tools import PythonREPLTool"
        },
    ]
    
    for tool_info in tools_info:
        print(f"   {tool_info['名称']}")
        print(f"   用途: {tool_info['用途']}")
        print(f"   导入: {tool_info['导入']}\n")


def demo_tool_binding():
    """演示工具绑定"""
    print("\n" + "=" * 60)
    print("工具与模型绑定")
    print("=" * 60)
    
    print("\n🔗 将工具绑定到模型:\n")
    
    print("""
   from langchain_openai import ChatOpenAI
   from langchain.agents import create_tool_calling_agent
   
   # 创建模型
   llm = ChatOpenAI(model="gpt-4")
   
   # 准备工具
   tools = [calculator, get_weather]
   
   # 绑定工具到模型
   llm_with_tools = llm.bind_tools(tools)
   
   # 或使用 Agent
   agent = create_tool_calling_agent(llm, tools, prompt)
   
   # 现在模型可以决定何时调用工具
   response = agent.invoke({"input": "北京天气怎么样？"})
   # 模型会自动调用 get_weather 工具
    """)


def demo_tool_best_practices():
    """工具最佳实践"""
    print("\n" + "=" * 60)
    print("工具开发最佳实践")
    print("=" * 60)
    
    print("""
💡 工具设计原则:

1. 清晰的命名
   - 使用动词开头: search_, calculate_, get_
   - 名称简洁明了

2. 详细的描述
   - 说明工具的功能
   - 说明使用场景
   - 说明返回值格式

3. 明确的参数
   - 使用 Pydantic 定义参数类型
   - 添加 Field 描述
   - 提供参数示例

4. 错误处理
   - 捕获异常并返回友好错误信息
   - 验证输入参数
   - 提供默认值

5. 安全性
   - 限制工具的执行范围
   - 避免执行危险操作
   - 对用户输入进行校验

示例:
    @tool
    def safe_calculator(expression: str) -> str:
        \"\"\"
        安全计算器，只支持基本数学运算。
        
        Args:
            expression: 数学表达式，如 "2 + 2"
            
        Returns:
            计算结果或错误信息
        \"\"\"
        try:
            # 安全检查
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "错误: 包含非法字符"
            
            result = eval(expression)
            return f"结果: {result}"
        except Exception as e:
            return f"错误: {str(e)}"
    """)


def main():
    demo_tool_decorator()
    demo_base_tool()
    demo_tool_list()
    demo_community_tools()
    demo_tool_binding()
    demo_tool_best_practices()
    
    print("\n" + "=" * 60)
    print("工具集成总结")
    print("=" * 60)
    print("""
💡 快速开始:
1. 使用 @tool 装饰器快速创建工具
2. 继承 BaseTool 创建复杂工具
3. 使用 bind_tools() 绑定到模型
4. 使用 Agent 自动调用工具

🔧 工具类型:
- @tool 装饰器: 简单函数工具
- BaseTool: 复杂自定义工具
- Community Tools: 预置工具

⚠️ 注意事项:
- 工具描述要清晰
- 参数类型要明确
- 做好错误处理
- 注意安全性
    """)
    
    print("\n✅ 工具集成学习完成！")


if __name__ == "__main__":
    main()
