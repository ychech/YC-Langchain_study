#!/usr/bin/env python3
"""
LangChain Community 核心概念 02 - 文本分割器 (Text Splitters)
功能：学习各种文本分割策略

核心概念：
- CharacterTextSplitter: 按字符分割
- RecursiveCharacterTextSplitter: 递归字符分割（推荐）
- TokenTextSplitter: 按 Token 分割
- MarkdownHeaderTextSplitter: Markdown 标题分割
"""
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def get_sample_text():
    """获取示例长文本"""
    return """
LangChain 是一个用于开发大语言模型应用程序的框架。

它提供了以下核心功能：

1. 模型调用
支持多种 LLM 和聊天模型，包括 OpenAI、Anthropic、本地模型等。
可以通过统一的接口调用不同的模型。

2. 提示词管理
提供模板化提示词，支持少样本学习。
可以动态填充变量，构建灵活的提示词。

3. 链式调用
将多个组件组合成复杂工作流。
使用 LCEL 语言，通过管道操作符连接组件。

4. 数据增强检索
RAG 实现，结合外部知识库。
支持向量检索、文档加载、文本分割等。

5. 智能代理
让模型自主决策使用工具。
支持 ReAct、Plan-and-Execute 等模式。

6. 记忆管理
支持短期和长期记忆。
可以保存对话历史，实现多轮对话。

LangChain 让开发 AI 应用变得简单高效。
开发者可以快速构建复杂的 LLM 应用。
"""


def demo_character_splitter():
    """演示字符分割器"""
    print("=" * 60)
    print("LangChain Community 核心概念 02 - 文本分割器")
    print("=" * 60)
    
    print("\n✂️ CharacterTextSplitter - 按字符分割\n")
    print("-" * 50)
    
    text = get_sample_text()
    
    # 基础分割
    print("\n1️⃣ 基础分割:")
    splitter = CharacterTextSplitter(
        separator="\n\n",      # 分隔符
        chunk_size=200,       # 每个块的最大字符数
        chunk_overlap=50,     # 块之间的重叠字符数
        length_function=len,  # 长度计算函数
        is_separator_regex=False
    )
    
    chunks = splitter.split_text(text)
    
    print(f"   原文长度: {len(text)} 字符")
    print(f"   分割成 {len(chunks)} 个块")
    print(f"\n   块预览:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk[:100]}...")


def demo_recursive_splitter():
    """演示递归分割器（推荐）"""
    print("\n" + "=" * 60)
    print("RecursiveCharacterTextSplitter - 递归分割（推荐）")
    print("=" * 60)
    
    print("\n🔪 递归分割会尝试多种分隔符，优先保持语义完整\n")
    
    text = get_sample_text()
    
    # 递归分割
    print("1️⃣ 递归分割:")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    
    print(f"   原文长度: {len(text)} 字符")
    print(f"   分割成 {len(chunks)} 个块")
    print(f"   平均块大小: {sum(len(c) for c in chunks) // len(chunks)} 字符")
    
    # 分割文档
    print("\n2️⃣ 分割 Document 对象:")
    doc = Document(page_content=text, metadata={"source": "example.txt"})
    doc_chunks = splitter.split_documents([doc])
    
    print(f"   1 个文档分割成 {len(doc_chunks)} 个块")
    print(f"   每个块保留元数据: {doc_chunks[0].metadata}")


def demo_token_splitter():
    """演示 Token 分割器"""
    print("\n" + "=" * 60)
    print("TokenTextSplitter - 按 Token 分割")
    print("=" * 60)
    
    print("\n🎫 按 Token 数量分割（适合 LLM 上下文限制）\n")
    
    text = get_sample_text()
    
    # 注意：需要 tiktoken
    print("1️⃣ Token 分割:")
    print("   使用 tiktoken 计算 token 数")
    
    try:
        splitter = TokenTextSplitter(
            chunk_size=50,      # 每个块的最大 token 数
            chunk_overlap=10    # 重叠 token 数
        )
        
        chunks = splitter.split_text(text)
        
        print(f"   分割成 {len(chunks)} 个块")
        print(f"\n   块预览:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   块 {i}: {chunk[:80]}...")
    except ImportError:
        print("   ⚠️ 需要安装 tiktoken: pip install tiktoken")


def demo_markdown_splitter():
    """演示 Markdown 分割器"""
    print("\n" + "=" * 60)
    print("MarkdownHeaderTextSplitter - Markdown 标题分割")
    print("=" * 60)
    
    print("\n📝 按 Markdown 标题层级分割\n")
    
    markdown_text = """
# LangChain 介绍

LangChain 是一个用于开发 LLM 应用的框架。

## 核心组件

### 模型接口
支持多种 LLM 和聊天模型。

### 提示词模板
提供灵活的提示词管理。

## 使用场景

### RAG 应用
结合外部知识库增强 LLM。

### Agent 应用
让 LLM 自主使用工具。
"""
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    chunks = splitter.split_text(markdown_text)
    
    print(f"   分割成 {len(chunks)} 个块\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i}:")
        print(f"     元数据: {chunk.metadata}")
        print(f"     内容: {chunk.page_content[:60]}...\n")


def demo_code_splitter():
    """演示代码分割器"""
    print("\n" + "=" * 60)
    print("代码分割器")
    print("=" * 60)
    
    print("\n💻 专门处理代码文件的分割器:\n")
    
    print("""
   from langchain_text_splitters import (
       PythonCodeTextSplitter,
       RecursiveCharacterTextSplitter
   )
   
   # Python 代码分割
   python_splitter = PythonCodeTextSplitter(
       chunk_size=1000,
       chunk_overlap=200
   )
   
   # 或使用递归分割器处理多种语言
   code_splitter = RecursiveCharacterTextSplitter.from_language(
       language="python",
       chunk_size=1000,
       chunk_overlap=200
   )
   
   # 支持的语言:
   # python, javascript, typescript, java, cpp, go, rust, etc.
    """)


def demo_splitter_comparison():
    """分割器对比"""
    print("\n" + "=" * 60)
    print("文本分割器对比")
    print("=" * 60)
    
    print("""
┌─────────────────────────┬─────────────────────────────┬──────────────────┐
│ 分割器                   │ 适用场景                     │ 特点             │
├─────────────────────────┼─────────────────────────────┼──────────────────┤
│ CharacterTextSplitter   │ 通用文本                     │ 简单直接         │
│ RecursiveCharacterText  │ 通用文本（推荐）              │ 保持语义完整     │
│ TokenTextSplitter       │ 需要精确控制 token 数         │ 基于 tokenizer   │
│ MarkdownHeaderText      │ Markdown 文档                │ 保留标题结构     │
│ PythonCodeTextSplitter  │ Python 代码                  │ 保留代码结构     │
└─────────────────────────┴─────────────────────────────┴──────────────────┘
    """)


def demo_best_practices():
    """最佳实践"""
    print("\n" + "=" * 60)
    print("分割参数选择建议")
    print("=" * 60)
    
    print("""
📊 参数选择:

1. chunk_size (块大小)
   - 一般文档: 500-1000 字符
   - 代码: 1000-2000 字符
   - 根据 LLM 上下文限制调整

2. chunk_overlap (重叠大小)
   - 建议: chunk_size 的 10-20%
   - 作用: 保持上下文连贯性
   - 太大: 冗余增加
   - 太小: 可能丢失上下文

3. 分隔符选择
   - 优先: 段落 (\\n\\n)
   - 其次: 句子 (。、！、？)
   - 最后: 单词 (空格)

🎯 推荐配置:
   RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50,
       separators=["\\n\\n", "\\n", "。", "，", " ", ""]
   )
    """)


def main():
    demo_character_splitter()
    demo_recursive_splitter()
    demo_token_splitter()
    demo_markdown_splitter()
    demo_code_splitter()
    demo_splitter_comparison()
    demo_best_practices()
    
    print("\n" + "=" * 60)
    print("文本分割器总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 通用文本使用 RecursiveCharacterTextSplitter
2. 根据 LLM 上下文限制选择 chunk_size
3. 设置适当的 chunk_overlap 保持连贯性
4. Markdown 使用 MarkdownHeaderTextSplitter
5. 代码使用语言特定的分割器

⚠️ 注意事项:
- 块太大会超出 LLM 上下文限制
- 块太小会丢失上下文信息
- 重叠太多会增加冗余
- 重叠太少会丢失连贯性
    """)
    
    print("\n✅ 文本分割器学习完成！")


if __name__ == "__main__":
    main()
