#!/usr/bin/env python3
"""
LangChain Core 核心概念 04 - 输出解析器 (Output Parsers)
功能：将模型输出解析为结构化数据

核心概念：
- StrOutputParser: 字符串输出
- JsonOutputParser: JSON 解析
- PydanticOutputParser: Pydantic 模型解析
- CommaSeparatedListOutputParser: 列表解析
"""
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def demo_str_parser():
    """演示字符串输出解析器"""
    print("=" * 60)
    print("LangChain Core 核心概念 04 - 输出解析器")
    print("=" * 60)
    
    print("\n📝 StrOutputParser - 字符串输出\n")
    print("-" * 50)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    # 创建解析器
    parser = StrOutputParser()
    
    # 调用模型
    response = llm.invoke("你好")
    
    # 解析输出
    parsed = parser.invoke(response)
    
    print(f"   原始响应类型: {type(response)}")
    print(f"   解析后类型: {type(parsed)}")
    print(f"   解析结果: {parsed}")
    
    # 在 LCEL 链中使用
    print("\n   在 LCEL 链中使用:")
    chain = llm | parser
    result = chain.invoke("什么是 Python？")
    print(f"   链式结果: {result[:50]}...")


def demo_list_parser():
    """演示列表输出解析器"""
    print("\n" + "=" * 60)
    print("CommaSeparatedListOutputParser - 列表解析")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    parser = CommaSeparatedListOutputParser()
    
    # 获取格式指令
    format_instructions = parser.get_format_instructions()
    
    print(f"\n📋 格式指令:\n   {format_instructions}\n")
    
    # 创建提示词模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个列表生成助手。{format_instructions}"),
        ("human", "{question}")
    ])
    
    # 部分填充格式指令
    prompt = template.partial(format_instructions=format_instructions)
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({"question": "列出 5 种编程语言"})
    
    print(f"📤 结果类型: {type(result)}")
    print(f"📤 结果内容:")
    for i, item in enumerate(result, 1):
        print(f"   {i}. {item}")


def demo_json_parser():
    """演示 JSON 输出解析器"""
    print("\n" + "=" * 60)
    print("JsonOutputParser - JSON 解析")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    parser = JsonOutputParser()
    
    # 创建提示词
    template = ChatPromptTemplate.from_messages([
        ("system", """你是一个 JSON 生成助手。请严格按照以下格式返回 JSON：
{{
    "name": "名称",
    "description": "描述",
    "tags": ["标签1", "标签2"]
}}"""),
        ("human", "请介绍 {topic}")
    ])
    
    # 构建链
    chain = template | llm | parser
    
    # 调用
    result = chain.invoke({"topic": "Python 编程语言"})
    
    print(f"\n📤 结果类型: {type(result)}")
    print(f"📤 结果内容:")
    for key, value in result.items():
        print(f"   {key}: {value}")


def demo_pydantic_parser():
    """演示 Pydantic 输出解析器"""
    print("\n" + "=" * 60)
    print("PydanticOutputParser - Pydantic 模型解析")
    print("=" * 60)
    
    # 定义 Pydantic 模型
    class Movie(BaseModel):
        """电影信息"""
        name: str = Field(description="电影名称")
        director: str = Field(description="导演")
        year: int = Field(description="上映年份")
        genres: list = Field(description="类型列表")
        rating: float = Field(description="评分 (0-10)")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    parser = PydanticOutputParser(pydantic_object=Movie)
    
    # 获取格式指令
    format_instructions = parser.get_format_instructions()
    
    print(f"\n📋 Pydantic 模型定义:\n")
    print(f"   class Movie(BaseModel):")
    print(f"       name: str")
    print(f"       director: str")
    print(f"       year: int")
    print(f"       genres: list")
    print(f"       rating: float")
    
    # 创建提示词
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个电影信息提取助手。{format_instructions}"),
        ("human", "{description}")
    ])
    
    prompt = template.partial(format_instructions=format_instructions)
    
    # 构建链
    chain = prompt | llm | parser
    
    # 调用
    description = """
    《肖申克的救赎》是一部1994年上映的美国剧情片，
    由弗兰克·德拉邦特执导，蒂姆·罗宾斯主演。
    这部电影在 IMDb 上获得了 9.3 分的高分，
    被誉为电影史上最伟大的作品之一。
    """
    
    result = chain.invoke({"description": description})
    
    print(f"\n📤 解析结果 (Pydantic 对象):\n")
    print(f"   类型: {type(result)}")
    print(f"   名称: {result.name}")
    print(f"   导演: {result.director}")
    print(f"   年份: {result.year}")
    print(f"   类型: {result.genres}")
    print(f"   评分: {result.rating}")


def demo_custom_parser():
    """演示自定义解析器"""
    print("\n" + "=" * 60)
    print("自定义解析器")
    print("=" * 60)
    
    from langchain_core.output_parsers import BaseOutputParser
    
    class BulletPointParser(BaseOutputParser):
        """自定义解析器：将输出解析为要点列表"""
        
        def parse(self, text: str) -> list:
            """解析文本为要点列表"""
            lines = text.strip().split('\n')
            points = []
            for line in lines:
                # 提取以 - 或 * 或数字开头的行
                stripped = line.strip()
                if stripped.startswith(('- ', '* ')):
                    points.append(stripped[2:])
                elif stripped.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
                    points.append(stripped[3:])
            return points
        
        @property
        def _type(self) -> str:
            return "bullet_point"
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    
    parser = BulletPointParser()
    
    # 构建链
    chain = llm | parser
    
    # 调用
    result = chain.invoke("列出 Python 的 5 个优点，使用要点格式")
    
    print(f"\n📤 自定义解析结果:\n")
    print(f"   类型: {type(result)}")
    print(f"   内容:")
    for i, point in enumerate(result, 1):
        print(f"   {i}. {point}")


def main():
    demo_str_parser()
    demo_list_parser()
    demo_json_parser()
    demo_pydantic_parser()
    demo_custom_parser()
    
    print("\n" + "=" * 60)
    print("输出解析器总结")
    print("=" * 60)
    print("""
┌─────────────────────────┬────────────────────────────────────┐
│ 解析器                   │ 用途                               │
├─────────────────────────┼────────────────────────────────────┤
│ StrOutputParser         │ 提取字符串内容                     │
│ JsonOutputParser        │ 解析 JSON 格式                     │
│ PydanticOutputParser    │ 解析为 Pydantic 对象               │
│ CommaSeparatedList      │ 解析逗号分隔的列表                 │
│ BaseOutputParser        │ 基类，用于自定义解析器             │
└─────────────────────────┴────────────────────────────────────┘

💡 最佳实践:
1. 简单文本使用 StrOutputParser
2. 结构化数据使用 PydanticOutputParser
3. 列表数据使用 CommaSeparatedListOutputParser
4. 复杂场景自定义解析器继承 BaseOutputParser
    """)
    
    print("\n✅ 输出解析器学习完成！")


if __name__ == "__main__":
    main()
