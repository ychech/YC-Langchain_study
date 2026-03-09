#!/usr/bin/env python3
"""
LangChain Community 核心概念 07 - 聊天模型 (Chat Models)
功能：学习集成各种 LLM 提供商

核心概念：
- OpenAI: GPT 系列模型
- Anthropic: Claude 系列模型
- 本地模型: Ollama, LlamaCpp
- 其他提供商: Azure, Bedrock 等
"""
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def demo_openai():
    """演示 OpenAI 模型"""
    print("=" * 60)
    print("LangChain Community 核心概念 07 - 聊天模型")
    print("=" * 60)
    
    print("\n🤖 OpenAI Chat Models\n")
    print("-" * 50)
    
    print("\n1️⃣ GPT-3.5 Turbo:")
    print("""
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI(
       model="gpt-3.5-turbo",
       api_key="your-api-key",
       temperature=0.7
   )
   response = llm.invoke("你好")
    """)
    
    print("\n2️⃣ GPT-4:")
    print("""
   llm = ChatOpenAI(
       model="gpt-4",
       api_key="your-api-key",
       temperature=0.7
   )
    """)
    
    print("\n3️⃣ 当前使用 (DeepSeek - OpenAI 兼容接口):")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    )
    
    response = llm.invoke("用一句话介绍你自己")
    print(f"   回复: {response.content}")


def demo_anthropic():
    """演示 Anthropic Claude"""
    print("\n" + "=" * 60)
    print("Anthropic Claude")
    print("=" * 60)
    
    print("\n🧠 Claude 系列模型:\n")
    
    print("""
   from langchain_anthropic import ChatAnthropic
   
   # Claude 3 Opus (最强)
   llm = ChatAnthropic(
       model="claude-3-opus-20240229",
       anthropic_api_key="your-api-key"
   )
   
   # Claude 3 Sonnet (平衡)
   llm = ChatAnthropic(
       model="claude-3-sonnet-20240229",
       anthropic_api_key="your-api-key"
   )
   
   # Claude 3 Haiku (最快)
   llm = ChatAnthropic(
       model="claude-3-haiku-20240307",
       anthropic_api_key="your-api-key"
   )
    """)


def demo_local_models():
    """演示本地模型"""
    print("\n" + "=" * 60)
    print("本地模型 (Ollama)")
    print("=" * 60)
    
    print("\n💻 使用 Ollama 运行本地模型:\n")
    
    print("""
   # 1. 安装 Ollama
   # https://ollama.ai/
   
   # 2. 拉取模型
   # ollama pull llama2
   # ollama pull mistral
   # ollama pull qwen
   
   # 3. 在 LangChain 中使用
   from langchain_community.chat_models import ChatOllama
   
   llm = ChatOllama(
       model="llama2",
       base_url="http://localhost:11434"
   )
   
   response = llm.invoke("你好")
    """)


def demo_azure_openai():
    """演示 Azure OpenAI"""
    print("\n" + "=" * 60)
    print("Azure OpenAI")
    print("=" * 60)
    
    print("\n☁️ Azure OpenAI 服务:\n")
    
    print("""
   from langchain_openai import AzureChatOpenAI
   
   llm = AzureChatOpenAI(
       azure_endpoint="https://your-resource.openai.azure.com/",
       azure_deployment="your-deployment-name",
       openai_api_version="2024-02-01",
       api_key="your-azure-api-key"
   )
    """)


def demo_other_providers():
    """演示其他提供商"""
    print("\n" + "=" * 60)
    print("其他 LLM 提供商")
    print("=" * 60)
    
    providers = [
        {
            "名称": "Google (Gemini)",
            "导入": "from langchain_google_genai import ChatGoogleGenerativeAI",
            "模型": "gemini-pro, gemini-pro-vision"
        },
        {
            "名称": "Cohere",
            "导入": "from langchain_cohere import ChatCohere",
            "模型": "command, command-r"
        },
        {
            "名称": "Mistral AI",
            "导入": "from langchain_mistralai import ChatMistralAI",
            "模型": "mistral-small, mistral-medium, mistral-large"
        },
        {
            "名称": "百度 (文心一言)",
            "导入": "from langchain_community.chat_models import QianfanChatEndpoint",
            "模型": "ERNIE-Bot, ERNIE-Bot-4"
        },
        {
            "名称": "阿里 (通义千问)",
            "导入": "from langchain_community.chat_models import Tongyi",
            "模型": "qwen-turbo, qwen-plus, qwen-max"
        },
        {
            "名称": "智谱 (ChatGLM)",
            "导入": "from langchain_community.chat_models import ChatZhipuAI",
            "模型": "chatglm_turbo, chatglm_pro"
        }
    ]
    
    for provider in providers:
        print(f"\n   {provider['名称']}")
        print(f"   导入: {provider['导入']}")
        print(f"   模型: {provider['模型']}")


def demo_model_comparison():
    """模型对比"""
    print("\n" + "=" * 60)
    print("模型选择参考")
    print("=" * 60)
    
    print("""
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ 提供商           │ 推荐模型         │ 特点            │ 适用场景        │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ OpenAI          │ gpt-4 / gpt-3.5 │ 能力强，生态好   │ 通用场景        │
│ Anthropic       │ claude-3-opus   │ 长文本，安全性   │ 文档处理        │
│ Google          │ gemini-pro      │ 多模态          │ 图文理解        │
│ 本地 (Ollama)   │ llama2 / mistral│ 隐私，免费      │ 本地部署        │
│ 百度            │ ERNIE-Bot-4     │ 中文好          │ 中文场景        │
│ 阿里            │ qwen-max        │ 中文好          │ 中文场景        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

选择建议:
- 英文通用: OpenAI GPT-4 / Claude-3
- 中文场景: 通义千问 / 文心一言 / Claude-3
- 成本控制: GPT-3.5 / Claude-3-Haiku
- 隐私敏感: 本地部署 (Ollama)
- 长文档: Claude-3 (200K 上下文)
    """)


def demo_unified_interface():
    """演示统一接口"""
    print("\n" + "=" * 60)
    print("统一接口的优势")
    print("=" * 60)
    
    print("\n🎯 LangChain 的统一接口:\n")
    
    print("   无论使用哪个提供商，接口都相同:\n")
    
    print("""
   # OpenAI
   openai_llm = ChatOpenAI(model="gpt-4")
   
   # Anthropic
   anthropic_llm = ChatAnthropic(model="claude-3-opus")
   
   # Ollama
   ollama_llm = ChatOllama(model="llama2")
   
   # 使用方式完全相同!
   for llm in [openai_llm, anthropic_llm, ollama_llm]:
       result = llm.invoke("你好")  # 相同接口
       print(result.content)
   
   # 可以轻松切换模型，无需修改业务代码
    """)


def main():
    demo_openai()
    demo_anthropic()
    demo_local_models()
    demo_azure_openai()
    demo_other_providers()
    demo_model_comparison()
    demo_unified_interface()
    
    print("\n" + "=" * 60)
    print("聊天模型总结")
    print("=" * 60)
    print("""
💡 快速开始:
1. 获取 API Key
2. 安装对应包: pip install langchain-openai
3. 创建模型实例
4. 使用统一接口调用

🔧 配置要点:
- model: 模型名称
- api_key: API 密钥
- base_url: 自定义端点 (可选)
- temperature: 创造性程度
- max_tokens: 最大输出长度

⚠️ 注意事项:
- 保护好 API Key
- 注意模型调用成本
- 处理 API 限制和错误
- 选择合适的模型版本
    """)
    
    print("\n✅ 聊天模型学习完成！")


if __name__ == "__main__":
    main()
