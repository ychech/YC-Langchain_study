#!/usr/bin/env python3
"""
LangChain Community 核心概念 01 - 文档加载器 (Document Loaders)
功能：学习加载各种格式的文档

核心概念：
- TextLoader: 加载文本文件
- PyPDFLoader: 加载 PDF 文件
- UnstructuredFileLoader: 加载各种格式
- DirectoryLoader: 批量加载目录
"""
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
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


def create_sample_files():
    """创建示例文件"""
    sample_dir = Path("sample_files")
    sample_dir.mkdir(exist_ok=True)
    
    # 创建示例文本文件
    txt_file = sample_dir / "sample.txt"
    txt_file.write_text("""
LangChain 是一个用于开发大语言模型应用的框架。

它提供了以下核心功能：
1. 模型调用：支持多种 LLM 和聊天模型
2. 提示词管理：模板化提示词，支持少样本学习
3. 链式调用：将多个组件组合成复杂工作流
4. 数据增强检索：RAG 实现，结合外部知识

LangChain 让开发 AI 应用变得简单高效。
""", encoding="utf-8")
    
    # 创建示例 CSV 文件
    csv_file = sample_dir / "sample.csv"
    csv_file.write_text("""name,age,city
张三,25,北京
李四,30,上海
王五,28,深圳
""", encoding="utf-8")
    
    print(f"✅ 示例文件已创建在: {sample_dir.absolute()}")
    return sample_dir


def demo_text_loader():
    """演示文本加载器"""
    print("=" * 60)
    print("LangChain Community 核心概念 01 - 文档加载器")
    print("=" * 60)
    
    print("\n📄 TextLoader - 加载文本文件\n")
    print("-" * 50)
    
    # 创建示例文件
    sample_dir = create_sample_files()
    txt_path = sample_dir / "sample.txt"
    
    # 方式 1: 基础加载
    print("\n1️⃣ 基础加载:")
    loader = TextLoader(str(txt_path), encoding="utf-8")
    documents = loader.load()
    
    print(f"   加载文档数: {len(documents)}")
    print(f"   文档内容长度: {len(documents[0].page_content)} 字符")
    print(f"   元数据: {documents[0].metadata}")
    
    # 方式 2: 懒加载（适合大文件）
    print("\n2️⃣ 懒加载 (lazy_load):")
    loader2 = TextLoader(str(txt_path), encoding="utf-8")
    
    doc_count = 0
    for doc in loader2.lazy_load():
        doc_count += 1
        print(f"   文档 {doc_count}: {len(doc.page_content)} 字符")
    
    # 显示内容预览
    print("\n3️⃣ 内容预览:")
    content = documents[0].page_content[:200]
    print(f"   {content}...")


def demo_csv_loader():
    """演示 CSV 加载器"""
    print("\n" + "=" * 60)
    print("CSVLoader - 加载 CSV 文件")
    print("=" * 60)
    
    sample_dir = Path("sample_files")
    csv_path = sample_dir / "sample.csv"
    
    print("\n📊 CSV 加载:\n")
    
    loader = CSVLoader(str(csv_path), encoding="utf-8")
    documents = loader.load()
    
    print(f"   加载行数: {len(documents)}")
    print(f"\n   内容预览:")
    
    for i, doc in enumerate(documents, 1):
        print(f"\n   行 {i}:")
        print(f"     内容: {doc.page_content}")
        print(f"     来源: {doc.metadata.get('source', 'N/A')}")
        print(f"     行号: {doc.metadata.get('row', 'N/A')}")


def demo_unstructured_loader():
    """演示 Unstructured 加载器"""
    print("\n" + "=" * 60)
    print("UnstructuredFileLoader - 通用文档加载")
    print("=" * 60)
    
    print("\n📑 Unstructured 支持多种格式:\n")
    print("   - PDF (.pdf)")
    print("   - Word (.doc, .docx)")
    print("   - PowerPoint (.ppt, .pptx)")
    print("   - HTML (.html, .htm)")
    print("   - 图片 (.jpg, .png) - OCR")
    print("   - 等等...")
    
    print("\n💡 使用示例:")
    print("""
   from langchain_community.document_loaders import UnstructuredFileLoader
   
   loader = UnstructuredFileLoader("document.pdf")
   docs = loader.load()
   
   # 或使用 UnstructuredPDFLoader 专门加载 PDF
   from langchain_community.document_loaders import PyPDFLoader
   loader = PyPDFLoader("document.pdf")
   docs = loader.load()
    """)


def demo_directory_loader():
    """演示目录加载器"""
    print("\n" + "=" * 60)
    print("DirectoryLoader - 批量加载目录")
    print("=" * 60)
    
    print("\n📁 批量加载整个目录:\n")
    
    print("""
   from langchain_community.document_loaders import DirectoryLoader
   
   # 加载目录下所有文本文件
   loader = DirectoryLoader(
       "./documents",           # 目录路径
       glob="**/*.txt",         # 文件匹配模式
       loader_cls=TextLoader,   # 使用的加载器
       show_progress=True       # 显示进度
   )
   
   docs = loader.load()
   print(f"加载了 {len(docs)} 个文档")
    """)
    
    # 实际演示
    from langchain_community.document_loaders import DirectoryLoader
    
    sample_dir = Path("sample_files")
    
    if sample_dir.exists():
        loader = DirectoryLoader(
            str(sample_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        try:
            docs = loader.load()
            print(f"\n   ✅ 实际加载: {len(docs)} 个文档")
            for doc in docs:
                print(f"      - {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"\n   ⚠️ 加载失败: {e}")


def demo_web_loader():
    """演示网页加载器"""
    print("\n" + "=" * 60)
    print("WebBaseLoader - 加载网页内容")
    print("=" * 60)
    
    print("\n🌐 加载网页内容:\n")
    
    print("""
   from langchain_community.document_loaders import WebBaseLoader
   
   # 加载单个网页
   loader = WebBaseLoader("https://python.langchain.com/")
   docs = loader.load()
   
   # 加载多个网页
   loader = WebBaseLoader([
       "https://python.langchain.com/",
       "https://docs.python.org/3/"
   ])
   docs = loader.load()
    """)


def demo_loader_comparison():
    """加载器对比"""
    print("\n" + "=" * 60)
    print("文档加载器对比")
    print("=" * 60)
    
    print("""
┌────────────────────────┬─────────────────────────────┬──────────────────┐
│ 加载器                  │ 适用格式                     │ 特点             │
├────────────────────────┼─────────────────────────────┼──────────────────┤
│ TextLoader             │ .txt, .md, .py 等文本        │ 简单高效         │
│ PyPDFLoader            │ .pdf                        │ 专门处理 PDF     │
│ CSVLoader              │ .csv                        │ 保留行元数据     │
│ UnstructuredFileLoader │ 多种格式                     │ 通用但较重       │
│ DirectoryLoader        │ 目录批量加载                 │ 支持递归和过滤   │
│ WebBaseLoader          │ 网页 URL                     │ 自动提取正文     │
│ JSONLoader             │ .json                       │ 支持 JSONPath    │
│ Docx2txtLoader         │ .docx                       │ Word 文档        │
└────────────────────────┴─────────────────────────────┴──────────────────┘
    """)


def main():
    demo_text_loader()
    demo_csv_loader()
    demo_unstructured_loader()
    demo_directory_loader()
    demo_web_loader()
    demo_loader_comparison()
    
    print("\n" + "=" * 60)
    print("文档加载器总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 文本文件使用 TextLoader
2. PDF 使用 PyPDFLoader
3. 批量加载使用 DirectoryLoader
4. 大文件使用 lazy_load()
5. 指定正确的 encoding (utf-8)
6. 检查加载后的 metadata

🎯 使用流程:
1. 选择合适的 Loader
2. 配置加载参数
3. 调用 load() 或 lazy_load()
4. 检查 Document 对象
5. 进行后续处理（分割、嵌入等）
    """)
    
    print("\n✅ 文档加载器学习完成！")


if __name__ == "__main__":
    main()
