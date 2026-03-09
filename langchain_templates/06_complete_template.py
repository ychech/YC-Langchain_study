#!/usr/bin/env python3
"""
LangChain 1.0.x 完整综合模板
功能：整合 LLM + Memory + RAG + Agent + LangSmith 追踪

注意：LangChain 1.0 使用全新的 API 设计
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

# ========== 配置 ==========
class Config:
    """应用配置"""
    MODEL = "deepseek-chat"
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = "https://api.deepseek.com/beta/v1"
    TEMPERATURE = 0.3
    MAX_TOKENS = 2048
    
    # LangSmith 配置
    LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")

# ========== 初始化 LLM ==========
llm = ChatOpenAI(
    model=Config.MODEL,
    api_key=Config.API_KEY,
    base_url=Config.BASE_URL,
    temperature=Config.TEMPERATURE,
    max_tokens=Config.MAX_TOKENS
)

# ========== 工具定义 ==========
@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取相关信息"""
    # 这里可以连接真实的知识库
    knowledge = {
        "langchain": "LangChain 1.0 于 2025年10月发布，是首个稳定版本。",
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"未找到关于 '{query}' 的知识"

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ========== RAG 系统 ==========
class RAGSystem:
    """检索增强生成系统"""
    
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        
    def create_from_texts(self, texts: List[str]):
        """从文本创建向量库"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # 创建文档
        documents = [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(texts)]
        
        # 分割
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        
        # 嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 构建向量库
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        return self.retriever
    
    def query(self, question: str) -> str:
        """查询"""
        if not self.retriever:
            return "知识库未初始化"
        
        docs = self.retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "基于以下上下文回答问题：\n\n{context}"),
            ("user", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

# ========== 对话系统 ==========
class ChatSystem:
    """带记忆的对话系统"""
    
    def __init__(self):
        self.conversations = {}
        self.max_history = 10  # 最多保留 10 条消息
    
    def get_history(self, session_id: str) -> List:
        """获取对话历史"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]
    
    def trim_history(self, messages):
        """修剪历史消息，保留最近的"""
        if len(messages) > self.max_history:
            return messages[-self.max_history:]
        return messages
    
    def chat(self, session_id: str, user_input: str) -> str:
        """进行对话"""
        history = self.get_history(session_id)
        
        # 添加用户消息
        history.append(HumanMessage(content=user_input))
        
        # 构建提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是友好的 AI 助手，记住对话上下文。"),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # 修剪消息并调用
        trimmed = self.trim_history(history)
        chain = prompt | llm
        response = chain.invoke({"messages": trimmed})
        
        # 保存 AI 回复
        history.append(AIMessage(content=response.content))
        
        return response.content

# ========== Agent 系统 ==========
class AgentSystem:
    """智能代理系统"""
    
    def __init__(self, tools):
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""你是智能助手，可以使用工具帮助用户。
可用工具：搜索知识库、计算器等。
请根据用户需求选择合适的工具。"""
        )
    
    def run(self, user_input: str) -> str:
        """运行 Agent"""
        response = self.agent.invoke({"input": user_input})
        # LangChain 1.0 返回 messages 列表
        messages = response["messages"]
        return messages[-1].content if messages else "无响应"

# ========== 主应用 ==========
class AIAssistant:
    """完整的 AI 助手应用"""
    
    def __init__(self):
        self.rag = RAGSystem()
        self.chat = ChatSystem()
        self.agent = None
        
    def setup_knowledge_base(self, texts: List[str]):
        """设置知识库"""
        self.rag.create_from_texts(texts)
        print(f"✅ 知识库已加载，共 {len(texts)} 个文档")
        
    def setup_agent(self):
        """设置 Agent"""
        tools = [search_knowledge, calculator]
        self.agent = AgentSystem(tools)
        print("✅ Agent 已初始化")
        
    def chat_mode(self, query: str, session_id: str = "default"):
        """对话模式"""
        return self.chat.chat(session_id, query)
    
    def rag_mode(self, query: str):
        """RAG 模式"""
        return self.rag.query(query)
    
    def agent_mode(self, query: str):
        """Agent 模式"""
        if not self.agent:
            return "Agent 未初始化"
        return self.agent.run(query)

# ========== 使用示例 ==========
def main():
    print("="*60)
    print("LangChain 1.0.x 完整综合模板")
    print("="*60)
    
    # 初始化助手
    assistant = AIAssistant()
    
    # 设置知识库
    assistant.setup_knowledge_base([
        "LangChain 是一个用于开发大语言模型应用的框架。",
        "DeepSeek 是一家中国 AI 公司，开发了强大的大语言模型。",
    ])
    
    # 设置 Agent
    assistant.setup_agent()
    
    # 测试各种模式
    print("\n===== 1. 对话模式 =====")
    print(f"用户：你好")
    print(f"AI：{assistant.chat_mode('你好', 'demo')}\n")
    
    print(f"用户：我叫王五")
    print(f"AI：{assistant.chat_mode('我叫王五', 'demo')}\n")
    
    print(f"用户：我叫什么？")
    print(f"AI：{assistant.chat_mode('我叫什么？', 'demo')}\n")
    
    print("===== 2. RAG 模式 =====")
    print(f"用户：LangChain 是什么？")
    print(f"AI：{assistant.rag_mode('LangChain 是什么？')}\n")
    
    print("===== 3. Agent 模式 =====")
    print(f"用户：搜索关于 LangChain 的信息")
    print(f"AI：{assistant.agent_mode('搜索关于 LangChain 的信息')}\n")
    
    print("✅ 演示完成！")
    if Config.LANGSMITH_TRACING:
        print(f"📊 查看追踪：https://smith.langchain.com")

if __name__ == "__main__":
    # 需要导入 MessagesPlaceholder
    from langchain_core.prompts import MessagesPlaceholder
    main()
