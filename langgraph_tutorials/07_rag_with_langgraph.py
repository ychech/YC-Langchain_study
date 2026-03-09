#!/usr/bin/env python3
"""
LangGraph 核心概念 07 - 使用 LangGraph 实现完整 RAG
功能：结合检索和生成的完整 RAG 系统

核心概念：
- 检索节点: 从向量库检索文档
- 生成节点: 基于检索结果生成回答
- 评估节点: 评估回答质量
- 循环优化: 不满意时重新检索
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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


# ========== 1. 定义 RAG 状态 ==========
class RAGState(TypedDict):
    """RAG 系统状态"""
    question: str           # 用户问题
    documents: List[Document]  # 检索到的文档
    generation: str         # 生成的回答
    iterations: int         # 迭代次数
    max_iterations: int     # 最大迭代次数
    should_continue: bool   # 是否继续优化


# ========== 2. 初始化组件 ==========
class RAGComponents:
    """RAG 组件容器"""
    
    def __init__(self):
        print("🔄 初始化 RAG 组件...")
        
        # LLM
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )
        
        # 嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 向量库（使用示例数据）
        self.vectorstore = self._create_sample_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        print("✅ RAG 组件初始化完成")
    
    def _create_sample_vectorstore(self):
        """创建示例向量库"""
        sample_docs = [
            Document(page_content="LangGraph 是 LangChain 的扩展，用于构建有状态的 AI 应用。"),
            Document(page_content="LangGraph 的核心概念包括 State、Node、Edge 和 Checkpoint。"),
            Document(page_content="State 是 LangGraph 中共享的数据容器，所有节点都可以访问和修改。"),
            Document(page_content="Node 是 LangGraph 中的执行单元，每个节点是一个 Python 函数。"),
            Document(page_content="Edge 连接不同的节点，控制工作流的执行顺序。"),
            Document(page_content="RAG (检索增强生成) 结合外部知识库来增强 LLM 的回答能力。"),
            Document(page_content="向量数据库用于存储文档的嵌入向量，支持相似度搜索。"),
            Document(page_content="LangChain 提供了多种文档加载器，支持 PDF、网页、数据库等数据源。"),
        ]
        
        return FAISS.from_documents(sample_docs, self.embeddings)


# 全局组件
rag_components = None


def get_components():
    """获取 RAG 组件（单例）"""
    global rag_components
    if rag_components is None:
        rag_components = RAGComponents()
    return rag_components


# ========== 3. 定义 RAG 节点 ==========
def retrieve(state: RAGState) -> RAGState:
    """
    检索节点：从向量库检索相关文档
    """
    print(f"\n🔍 [Retrieve] 检索问题: {state['question']}")
    
    components = get_components()
    
    # 检索文档
    docs = components.retriever.invoke(state["question"])
    
    print(f"   检索到 {len(docs)} 个文档:")
    for i, doc in enumerate(docs, 1):
        print(f"   {i}. {doc.page_content[:50]}...")
    
    return {
        **state,
        "documents": docs,
        "iterations": state.get("iterations", 0) + 1
    }


def generate(state: RAGState) -> RAGState:
    """
    生成节点：基于检索的文档生成回答
    """
    print(f"\n✍️  [Generate] 生成回答...")
    
    components = get_components()
    
    # 构建上下文
    context = "\n\n".join([
        f"文档 {i+1}: {doc.page_content}"
        for i, doc in enumerate(state["documents"])
    ])
    
    # 生成提示词
    prompt = f"""基于以下检索到的文档，回答用户的问题。

检索到的文档:
{context}

用户问题: {state['question']}

要求:
1. 仅基于提供的文档回答
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁准确

回答:"""
    
    response = components.llm.invoke(prompt)
    
    print(f"   生成完成，长度: {len(response.content)} 字符")
    
    return {
        **state,
        "generation": response.content
    }


def evaluate(state: RAGState) -> RAGState:
    """
    评估节点：评估回答质量
    """
    print(f"\n🔍 [Evaluate] 评估回答质量...")
    
    components = get_components()
    
    # 简单的评估：检查回答是否包含"不知道"或"未找到"等词
    generation = state["generation"].lower()
    
    # 评估标准
    has_relevant_info = not any(
        phrase in generation
        for phrase in ["不知道", "未找到", "没有相关信息", "无法回答"]
    )
    
    # 检查迭代次数
    iterations = state.get("iterations", 0)
    max_iter = state.get("max_iterations", 3)
    
    should_continue = has_relevant_info or iterations >= max_iter
    
    if has_relevant_info:
        print(f"   ✅ 回答质量良好，可以输出")
    elif iterations >= max_iter:
        print(f"   ⚠️ 达到最大迭代次数 ({max_iter})，停止优化")
    else:
        print(f"   🔄 回答质量不佳，需要重新检索")
    
    return {
        **state,
        "should_continue": should_continue
    }


def rewrite_query(state: RAGState) -> RAGState:
    """
    查询重写节点：优化查询以获得更好的检索结果
    """
    print(f"\n🔄 [Rewrite] 重写查询...")
    
    components = get_components()
    
    prompt = f"""原始查询没有获得满意的回答。请重写查询，使其更具体、更容易检索到相关信息。

原始查询: {state['question']}

请生成一个更具体的查询:"""
    
    response = components.llm.invoke(prompt)
    new_question = response.content.strip()
    
    print(f"   重写前: {state['question']}")
    print(f"   重写后: {new_question}")
    
    return {
        **state,
        "question": new_question
    }


# ========== 4. 条件路由 ==========
def route_after_evaluate(state: RAGState) -> str:
    """评估后路由"""
    if state.get("should_continue", True):
        return "end"
    return "rewrite"


# ========== 5. 构建 RAG 图 ==========
def build_rag_graph():
    """构建 RAG 图"""
    
    workflow = StateGraph(RAGState)
    
    # 添加节点
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("rewrite", rewrite_query)
    
    # 设置入口
    workflow.set_entry_point("retrieve")
    
    # 添加边
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    
    # 条件边：评估后决定是结束还是重写
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "end": END,
            "rewrite": "rewrite"
        }
    )
    
    # 重写后重新检索
    workflow.add_edge("rewrite", "retrieve")
    
    return workflow.compile()


# ========== 6. 运行 RAG ==========
def main():
    print("=" * 60)
    print("LangGraph 核心概念 07 - RAG 完整实现")
    print("=" * 60)
    
    graph = build_rag_graph()
    
    # 测试问题
    test_questions = [
        "什么是 LangGraph？",
        "RAG 是什么？",
        "LangGraph 和 LangChain 有什么关系？",
    ]
    
    for question in test_questions:
        print("\n" + "="*60)
        print(f"问题: {question}")
        print("="*60)
        
        state = graph.invoke({
            "question": question,
            "documents": [],
            "generation": "",
            "iterations": 0,
            "max_iterations": 3,
            "should_continue": False
        })
        
        print(f"\n📤 最终回答:")
        print(f"{state['generation']}")
        print(f"\n📊 迭代次数: {state['iterations']}")
    
    # 展示 RAG 架构
    print("\n" + "="*60)
    print("RAG 系统架构")
    print("="*60)
    print("""
    ┌─────────────┐
    │   用户问题   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   检索节点   │◄────┐
    │  (Retrieve)  │     │
    └──────┬──────┘     │
           ▼            │
    ┌─────────────┐     │
    │   生成节点   │     │
    │  (Generate)  │     │
    └──────┬──────┘     │
           ▼            │
    ┌─────────────┐     │
    │   评估节点   │     │
    │  (Evaluate)  │     │
    └──────┬──────┘     │
           │            │
      质量不好?         │
           │            │
     是 ◄──┴──► 否      │
     │            │     │
     ▼            ▼     │
┌─────────┐   ┌────────┐│
│ 重写查询 │   │ 输出结果││
│(Rewrite)│   │  (END) ││
└────┬────┘   └────────┘│
     │                  │
     └──────────────────┘
    """)


if __name__ == "__main__":
    main()
    print("\n✅ RAG 系统测试完成！")
