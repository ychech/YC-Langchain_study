#!/usr/bin/env python3
"""
LangChain Community 核心概念 08 - 实用工具 (Utilities)
功能：学习各种实用工具和辅助功能

核心概念：
- SerpAPI: Google 搜索
- Arxiv: 学术论文查询
- Wikipedia: 百科查询
- GraphQL: 图数据库查询
"""
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
load_dotenv()


def demo_serpapi():
    """演示 SerpAPI 搜索"""
    print("=" * 60)
    print("LangChain Community 核心概念 08 - 实用工具")
    print("=" * 60)
    
    print("\n🔍 SerpAPI - Google 搜索\n")
    print("-" * 50)
    
    print("""
   from langchain_community.utilities import SerpAPIWrapper
   
   # 需要 SERPAPI_API_KEY
   search = SerpAPIWrapper()
   
   # 执行搜索
   result = search.run("Python 编程语言")
   print(result)
   
   # 返回搜索结果摘要
    """)


def demo_arxiv():
    """演示 Arxiv 论文查询"""
    print("\n" + "=" * 60)
    print("Arxiv - 学术论文")
    print("=" * 60)
    
    print("\n📄 查询学术论文:\n")
    
    print("""
   from langchain_community.utilities import ArxivAPIWrapper
   
   arxiv = ArxivAPIWrapper(
       top_k_results=3,
       doc_content_chars_max=1000
   )
   
   # 搜索论文
   result = arxiv.run("transformer architecture")
   print(result)
   
   # 返回论文标题、摘要、链接
    """)


def demo_wikipedia():
    """演示 Wikipedia 查询"""
    print("\n" + "=" * 60)
    print("Wikipedia - 百科查询")
    print("=" * 60)
    
    print("\n📚 查询百科知识:\n")
    
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
        
        wiki = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        )
        
        result = wiki.run("Python 编程语言")
        
        print("   查询: 'Python 编程语言'")
        print(f"   结果:\n{result[:300]}...")
        
    except Exception as e:
        print(f"   演示代码:\n")
        print("""
   from langchain_community.utilities import WikipediaAPIWrapper
   
   wiki = WikipediaAPIWrapper(
       top_k_results=2,
       doc_content_chars_max=500
   )
   
   result = wiki.run("Python 编程语言")
   print(result)
        """)


def demo_requests():
    """演示 HTTP 请求工具"""
    print("\n" + "=" * 60)
    print("TextRequestsWrapper - HTTP 请求")
    print("=" * 60)
    
    print("\n🌐 发送 HTTP 请求:\n")
    
    print("""
   from langchain_community.utilities import TextRequestsWrapper
   
   requests = TextRequestsWrapper()
   
   # GET 请求
   response = requests.get("https://api.github.com")
   
   # POST 请求
   response = requests.post(
       "https://httpbin.org/post",
       data={"key": "value"}
   )
    """)


def demo_openweathermap():
    """演示天气查询"""
    print("\n" + "=" * 60)
    print("OpenWeatherMap - 天气查询")
    print("=" * 60)
    
    print("\n🌤️ 查询天气信息:\n")
    
    print("""
   from langchain_community.utilities import OpenWeatherMapAPIWrapper
   
   weather = OpenWeatherMapAPIWrapper(
       openweathermap_api_key="your-api-key"
   )
   
   # 查询天气
   result = weather.run("北京")
   print(result)
    """)


def demo_graphql():
    """演示 GraphQL"""
    print("\n" + "=" * 60)
    print("GraphQL - 图数据库查询")
    print("=" * 60)
    
    print("\n📊 查询 GraphQL API:\n")
    
    print("""
   from langchain_community.utilities import GraphQLAPIWrapper
   
   graphql = GraphQLAPIWrapper(
       graphql_endpoint="https://api.github.com/graphql",
       custom_headers={"Authorization": "Bearer token"}
   )
   
   query = \"\"\"
   query {
       viewer {
           login
           name
       }
   }
   \"\"\"
   
   result = graphql.run(query)
    """)


def demo_google_search():
    """演示 Google 搜索"""
    print("\n" + "=" * 60)
    print("Google Search API")
    print("=" * 60)
    
    print("\n🔎 Google 搜索:\n")
    
    print("""
   # 方式 1: Google Custom Search
   from langchain_community.utilities import GoogleSearchAPIWrapper
   
   search = GoogleSearchAPIWrapper(
       google_api_key="your-api-key",
       google_cse_id="your-cse-id"
   )
   
   result = search.run("Python 教程")
   
   # 方式 2: SerpAPI (推荐)
   from langchain_community.utilities import SerpAPIWrapper
   
   search = SerpAPIWrapper(serpapi_api_key="your-key")
   result = search.run("Python 教程")
    """)


def show_utilities_comparison():
    """工具对比"""
    print("\n" + "=" * 60)
    print("实用工具汇总")
    print("=" * 60)
    
    print("""
┌─────────────────────────┬─────────────────────────────┬──────────────────┐
│ 工具                     │ 用途                        │ 需要 API Key     │
├─────────────────────────┼─────────────────────────────┼──────────────────┤
│ SerpAPIWrapper          │ Google 搜索                 │ 是               │
│ GoogleSearchAPIWrapper  │ Google Custom Search        │ 是               │
│ ArxivAPIWrapper         │ 学术论文查询                 │ 否               │
│ WikipediaAPIWrapper     │ 百科查询                    │ 否               │
│ OpenWeatherMapAPIWrapper│ 天气查询                    │ 是               │
│ TextRequestsWrapper     │ HTTP 请求                   │ 否               │
│ GraphQLAPIWrapper       │ GraphQL 查询                │ 视情况           │
│ DuckDuckGoSearchAPIWrapper│ DuckDuckGo 搜索           │ 否               │
└─────────────────────────┴─────────────────────────────┴──────────────────┘
    """)


def demo_integration_with_tools():
    """演示与工具集成"""
    print("\n" + "=" * 60)
    print("与 LangChain 工具集成")
    print("=" * 60)
    
    print("\n🔧 Utilities 可以转换为 Tools:\n")
    
    print("""
   from langchain_community.utilities import SerpAPIWrapper
   from langchain_core.tools import Tool
   
   # 创建 Utility
   search = SerpAPIWrapper()
   
   # 转换为 Tool
   search_tool = Tool(
       name="web_search",
       func=search.run,
       description="使用 Google 搜索网络信息"
   )
   
   # 在 Agent 中使用
   from langchain_classic.agents import create_tool_calling_agent
   
   tools = [search_tool]
   agent = create_tool_calling_agent(llm, tools, prompt)
    """)


def main():
    demo_serpapi()
    demo_arxiv()
    demo_wikipedia()
    demo_requests()
    demo_openweathermap()
    demo_graphql()
    demo_google_search()
    show_utilities_comparison()
    demo_integration_with_tools()
    
    print("\n" + "=" * 60)
    print("实用工具总结")
    print("=" * 60)
    print("""
💡 使用建议:
1. 搜索类: SerpAPI (Google) 或 DuckDuckGo (免费)
2. 学术类: ArxivAPIWrapper
3. 知识类: WikipediaAPIWrapper
4. 数据类: 根据 API 选择对应 Wrapper

🔧 使用流程:
1. 获取 API Key (如需要)
2. 导入对应的 Wrapper
3. 创建实例并配置
4. 调用 run() 方法
5. 处理返回结果

⚠️ 注意事项:
- 注意 API 调用限制
- 保护好 API Key
- 处理 API 错误
- 缓存结果避免重复调用
    """)
    
    print("\n✅ 实用工具学习完成！")


if __name__ == "__main__":
    main()
