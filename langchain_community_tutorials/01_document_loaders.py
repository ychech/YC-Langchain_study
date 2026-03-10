#!/usr/bin/env python3
"""
LangChain Community 核心概念 01 - 文档加载器 (Document Loaders)
功能：学习加载各种格式的文档

核心概念：
- TextLoader: 加载文本文件
- PyPDFLoader: 加载 PDF 文件
- UnstructuredFileLoader: 加载各种格式
- DirectoryLoader: 批量加载目录

文档加载器的作用：
在 LangChain 中，文档加载器（Document Loaders）是数据处理的起点，负责将各种格式的
文件读取并转换为 LangChain 的 Document 对象。Document 对象是 LangChain 中处理
文本数据的基本单元，包含文本内容（page_content）和元数据（metadata）两部分。

为什么需要文档加载器？
1. 统一接口：不同的文件格式有不同的读取方式，加载器提供统一接口
2. 元数据保留：自动记录文件来源、页码、行号等信息，便于追溯
3. 大文件支持：支持懒加载（lazy loading），避免内存溢出
4. 批量处理：支持整个目录的批量加载
"""

# =============================================================================
# 导入必要的模块
# =============================================================================

# 从 langchain_community 导入文档加载器
# langchain_community 包含社区贡献的各种工具，包括文档加载器
from langchain_community.document_loaders import (
    TextLoader,           # 文本文件加载器，用于加载 .txt、.md、.py 等纯文本文件
    CSVLoader,            # CSV 文件加载器，将 CSV 的每一行转换为一个 Document
    UnstructuredFileLoader  # 通用文件加载器，支持多种格式（PDF、Word、PPT 等）
)

# 从 langchain_core 导入 Document 类
# Document 是 LangChain 中文档的基本数据结构，包含 page_content 和 metadata
from langchain_core.documents import Document

# 用于加载环境变量（如 API 密钥等配置）
from dotenv import load_dotenv

# Python 标准库
import os       # 操作系统接口，用于环境变量操作
from pathlib import Path  # 面向对象的路径操作，比 os.path 更现代、更易用


# =============================================================================
# 环境变量配置
# =============================================================================

# 加载环境变量
# 首先尝试加载父目录中的 .env 文件，如果不存在则加载当前目录的
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


# =============================================================================
# 辅助函数：创建示例文件
# =============================================================================

def create_sample_files():
    """
    创建示例文件
    
    功能说明：
    这个函数用于创建教程所需的示例文件，包括文本文件和 CSV 文件。
    这样学习者可以直接运行代码，无需准备外部文件。
    
    返回值：
        Path: 示例文件所在的目录路径
    
    最佳实践：
    - 使用 Path 对象处理文件路径，比字符串拼接更安全和跨平台
    - 使用 mkdir(exist_ok=True) 避免目录已存在时报错
    - 显式指定 encoding="utf-8" 确保中文字符正确处理
    """
    # 创建示例文件存放目录
    # Path("sample_files") 创建相对路径对象，指向当前工作目录下的 sample_files 文件夹
    sample_dir = Path("sample_files")
    
    # mkdir() 创建目录，exist_ok=True 表示如果目录已存在则不报错
    sample_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 创建示例文本文件
    # -------------------------------------------------------------------------
    txt_file = sample_dir / "sample.txt"
    # write_text() 方法将字符串写入文件，自动处理文件打开和关闭
    txt_file.write_text("""
LangChain 是一个用于开发大语言模型应用的框架。

它提供了以下核心功能：
1. 模型调用：支持多种 LLM 和聊天模型
2. 提示词管理：模板化提示词，支持少样本学习
3. 链式调用：将多个组件组合成复杂工作流
4. 数据增强检索：RAG 实现，结合外部知识

LangChain 让开发 AI 应用变得简单高效。
""", encoding="utf-8")
    
    # -------------------------------------------------------------------------
    # 创建示例 CSV 文件
    # -------------------------------------------------------------------------
    csv_file = sample_dir / "sample.csv"
    csv_file.write_text("""name,age,city
张三,25,北京
李四,30,上海
王五,28,深圳
""", encoding="utf-8")
    
    # 打印创建成功的信息，absolute() 返回绝对路径
    print(f"✅ 示例文件已创建在: {sample_dir.absolute()}")
    return sample_dir


# =============================================================================
# 演示 1：TextLoader - 文本文件加载器
# =============================================================================

def demo_text_loader():
    """
    演示文本加载器的使用
    
    TextLoader 是最基础的文档加载器，用于加载纯文本文件。
    
    适用场景：
    - .txt 文本文件
    - .md Markdown 文件
    - .py Python 源代码
    - .json JSON 文件
    - 任何纯文本格式的文件
    
    核心方法：
    1. load() - 立即加载所有内容，返回 Document 列表
    2. lazy_load() - 惰性加载，返回生成器，适合大文件
    """
    # 打印标题
    print("=" * 60)
    print("LangChain Community 核心概念 01 - 文档加载器")
    print("=" * 60)
    
    print("\n📄 TextLoader - 加载文本文件\n")
    print("-" * 50)
    
    # 调用辅助函数创建示例文件
    sample_dir = create_sample_files()
    txt_path = sample_dir / "sample.txt"
    
    # -------------------------------------------------------------------------
    # 方式 1: 基础加载（load 方法）
    # -------------------------------------------------------------------------
    print("\n1️⃣ 基础加载:")
    
    # 创建 TextLoader 实例
    # 参数说明：
    # - file_path: 文件路径（字符串或 Path 对象）
    # - encoding: 文件编码，中文文件建议使用 "utf-8"
    loader = TextLoader(str(txt_path), encoding="utf-8")
    
    # load() 方法读取整个文件，返回 Document 对象列表
    # 对于 TextLoader，通常返回包含一个 Document 的列表
    documents = loader.load()
    
    # Document 对象包含两个主要属性：
    # - page_content: 文档的文本内容（字符串）
    # - metadata: 元数据字典，包含 source（文件路径）等信息
    print(f"   加载文档数: {len(documents)}")
    print(f"   文档内容长度: {len(documents[0].page_content)} 字符")
    print(f"   元数据: {documents[0].metadata}")
    
    # -------------------------------------------------------------------------
    # 方式 2: 懒加载（lazy_load 方法）
    # -------------------------------------------------------------------------
    print("\n2️⃣ 懒加载 (lazy_load):")
    
    # 懒加载适用于大文件，它不会一次性将所有内容加载到内存
    # 而是返回一个生成器，每次只处理一个文档
    loader2 = TextLoader(str(txt_path), encoding="utf-8")
    
    doc_count = 0
    # 使用 for 循环遍历生成器，逐个处理文档
    for doc in loader2.lazy_load():
        doc_count += 1
        print(f"   文档 {doc_count}: {len(doc.page_content)} 字符")
    
    # -------------------------------------------------------------------------
    # 显示内容预览
    # -------------------------------------------------------------------------
    print("\n3️⃣ 内容预览:")
    # 截取前 200 个字符作为预览
    content = documents[0].page_content[:200]
    print(f"   {content}...")


# =============================================================================
# 演示 2：CSVLoader - CSV 文件加载器
# =============================================================================

def demo_csv_loader():
    """
    演示 CSV 加载器的使用
    
    CSVLoader 专门用于加载 CSV 文件，将每一行转换为一个 Document 对象。
    
    特点：
    - 自动将 CSV 行转换为文本格式
    - 保留行号（row）和来源（source）等元数据
    - 支持自定义列的拼接方式
    
    适用场景：
    - 结构化数据的加载
    - 表格数据的 RAG 应用
    - 需要将 CSV 数据用于语义搜索
    """
    print("\n" + "=" * 60)
    print("CSVLoader - 加载 CSV 文件")
    print("=" * 60)
    
    # 指定 CSV 文件路径
    sample_dir = Path("sample_files")
    csv_path = sample_dir / "sample.csv"
    
    print("\n📊 CSV 加载:\n")
    
    # 创建 CSVLoader 实例
    # 参数说明：
    # - file_path: CSV 文件路径
    # - encoding: 文件编码
    # - csv_args: 传递给 csv.DictReader 的参数（可选）
    loader = CSVLoader(str(csv_path), encoding="utf-8")
    
    # load() 方法返回 Document 列表，每个 Document 对应 CSV 的一行
    documents = loader.load()
    
    print(f"   加载行数: {len(documents)}")
    print(f"\n   内容预览:")
    
    # 遍历每个 Document，显示内容和元数据
    for i, doc in enumerate(documents, 1):
        print(f"\n   行 {i}:")
        print(f"     内容: {doc.page_content}")
        # metadata 字典包含以下常用键：
        # - source: 文件路径
        # - row: 行号（从 0 开始）
        print(f"     来源: {doc.metadata.get('source', 'N/A')}")
        print(f"     行号: {doc.metadata.get('row', 'N/A')}")


# =============================================================================
# 演示 3：UnstructuredFileLoader - 通用文档加载器
# =============================================================================

def demo_unstructured_loader():
    """
    演示 Unstructured 加载器的使用
    
    Unstructured 是一个强大的 Python 库，可以解析多种文档格式。
    UnstructuredFileLoader 是 LangChain 对 Unstructured 的封装。
    
    支持的格式：
    - PDF 文档 (.pdf)
    - Word 文档 (.doc, .docx)
    - PowerPoint (.ppt, .pptx)
    - Excel 表格 (.xls, .xlsx)
    - HTML 网页 (.html, .htm)
    - 图片文件 (.jpg, .png) - 需要 OCR 支持
    - EPUB 电子书
    - 以及更多...
    
    安装依赖：
    pip install unstructured
    # 根据需要的格式，可能还需要安装额外的依赖：
    # pip install unstructured[pdf]      # PDF 支持
    # pip install unstructured[docx]     # Word 支持
    
    注意事项：
    - Unstructured 功能强大但依赖较多，安装包较大
    - 对于特定格式，建议使用专门的加载器（如 PyPDFLoader）
    """
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
    
    # 显示使用示例代码
    print("\n💡 使用示例:")
    print("""
   from langchain_community.document_loaders import UnstructuredFileLoader
   
   # 创建加载器实例
   loader = UnstructuredFileLoader("document.pdf")
   
   # 加载文档
   docs = loader.load()
   
   # 或使用 PyPDFLoader 专门加载 PDF（更轻量）
   from langchain_community.document_loaders import PyPDFLoader
   loader = PyPDFLoader("document.pdf")
   docs = loader.load()
    """)


# =============================================================================
# 演示 4：DirectoryLoader - 目录批量加载器
# =============================================================================

def demo_directory_loader():
    """
    演示目录加载器的使用
    
    DirectoryLoader 用于批量加载整个目录中的文件，是处理大量文档的利器。
    
    核心功能：
    - 递归遍历子目录
    - 支持文件过滤（glob 模式）
    - 支持多种加载器
    - 显示加载进度
    
    适用场景：
    - 批量处理文档库
    - 构建知识库
    - 大规模数据处理
    
    参数说明：
    - path: 目录路径
    - glob: 文件匹配模式，如 "**/*.txt" 表示所有子目录中的 .txt 文件
    - loader_cls: 使用的加载器类
    - loader_kwargs: 传递给加载器的参数
    - show_progress: 是否显示进度条
    - use_multithreading: 是否使用多线程加速
    """
    print("\n" + "=" * 60)
    print("DirectoryLoader - 批量加载目录")
    print("=" * 60)
    
    print("\n📁 批量加载整个目录:\n")
    
    # 显示代码示例
    print("""
   from langchain_community.document_loaders import DirectoryLoader
   
   # 加载目录下所有文本文件
   loader = DirectoryLoader(
       "./documents",           # 目录路径
       glob="**/*.txt",         # 文件匹配模式（递归所有子目录）
       loader_cls=TextLoader,   # 使用的加载器类
       show_progress=True       # 显示加载进度
   )
   
   docs = loader.load()
   print(f"加载了 {len(docs)} 个文档")
    """)
    
    # 实际演示
    from langchain_community.document_loaders import DirectoryLoader
    
    sample_dir = Path("sample_files")
    
    # 检查目录是否存在
    if sample_dir.exists():
        # 创建 DirectoryLoader 实例
        loader = DirectoryLoader(
            str(sample_dir),              # 目录路径（需要字符串）
            glob="**/*.txt",              # 匹配所有 .txt 文件
            loader_cls=TextLoader,        # 使用 TextLoader 加载每个文件
            loader_kwargs={"encoding": "utf-8"}  # 传递给 TextLoader 的参数
        )
        
        try:
            # 执行批量加载
            docs = loader.load()
            print(f"\n   ✅ 实际加载: {len(docs)} 个文档")
            # 显示每个加载文档的来源
            for doc in docs:
                print(f"      - {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"\n   ⚠️ 加载失败: {e}")


# =============================================================================
# 演示 5：WebBaseLoader - 网页加载器
# =============================================================================

def demo_web_loader():
    """
    演示网页加载器的使用
    
    WebBaseLoader 用于从网页 URL 加载内容，自动提取网页正文。
    
    核心功能：
    - 自动提取网页正文内容（去除导航、广告等）
    - 支持多个 URL 批量加载
    - 支持自定义请求头（模拟浏览器）
    
    适用场景：
    - 从网页获取最新信息
    - 构建基于网页内容的 RAG 应用
    - 网页内容分析和摘要
    
    注意事项：
    - 需要设置合适的 User-Agent，避免被网站拦截
    - 尊重网站的 robots.txt 和爬虫政策
    - 考虑添加适当的延迟，避免频繁请求
    """
    import os
    
    # 设置 User-Agent 环境变量，避免某些网站的拦截
    # User-Agent 告诉服务器请求来自什么客户端
    os.environ["USER_AGENT"] = "LangChain-Tutorial/1.0 (Python/3.11)"
    
    print("\n" + "=" * 60)
    print("WebBaseLoader - 加载网页内容")
    print("=" * 60)
    
    print("\n🌐 加载网页内容:\n")
    
    # 导入网页加载器
    from langchain_community.document_loaders import WebBaseLoader
    
    # 使用一个简单稳定的测试网页
    # httpbin.org 是一个用于测试 HTTP 请求的服务
    url = "https://httpbin.org/html"
    
    print(f"1️⃣ 加载单个网页:")
    print(f"   URL: {url}")
    
    try:
        # 创建 WebBaseLoader 实例，配置请求头模拟浏览器
        loader = WebBaseLoader(
            url,
            header_template={
                # User-Agent 模拟 Chrome 浏览器，提高兼容性
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        # 加载网页内容
        docs = loader.load()
        
        print(f"   ✅ 成功加载 {len(docs)} 个文档")
        if docs:
            doc = docs[0]
            print(f"   内容长度: {len(doc.page_content)} 字符")
            print(f"   内容预览:")
            print(f"   {doc.page_content[:300]}...")
            print(f"   元数据: {doc.metadata}")
    except Exception as e:
        print(f"   ⚠️ 加载失败: {e}")
    
    # 显示更多使用示例
    print("\n2️⃣ 代码示例:")
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
   
   # 配置请求头（模拟浏览器，避免被拦截）
   loader = WebBaseLoader(
       "https://example.com/",
       header_template={
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
       }
   )
    """)


# =============================================================================
# 演示 6：加载器对比总结
# =============================================================================

def demo_loader_comparison():
    """
    各种文档加载器的对比总结
    
    帮助学习者根据需求选择合适的加载器
    """
    print("\n" + "=" * 60)
    print("文档加载器对比")
    print("=" * 60)
    
    # 使用表格形式展示各种加载器的特点
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


# =============================================================================
# 主函数：程序入口
# =============================================================================

def main():
    """
    主函数：按顺序执行所有演示
    
    运行流程：
    1. 文本文件加载演示
    2. CSV 文件加载演示
    3. Unstructured 加载器介绍
    4. 目录批量加载演示
    5. 网页加载演示
    6. 加载器对比总结
    """
    # 依次调用各个演示函数
    demo_text_loader()
    demo_csv_loader()
    demo_unstructured_loader()
    demo_directory_loader()
    demo_web_loader()
    demo_loader_comparison()
    
    # 打印总结信息
    print("\n" + "=" * 60)
    print("文档加载器总结")
    print("=" * 60)
    print("""
💡 最佳实践:
1. 文本文件使用 TextLoader
   - 简单、轻量、高效
   - 记得指定 encoding="utf-8"

2. PDF 使用 PyPDFLoader
   - 专门优化，比 Unstructured 更轻量
   - 支持按页分割

3. 批量加载使用 DirectoryLoader
   - 支持递归和文件过滤
   - 可显示加载进度

4. 大文件使用 lazy_load()
   - 避免内存溢出
   - 适合流式处理

5. 指定正确的 encoding
   - 中文文件使用 "utf-8"
   - Windows 文件可能需要 "gbk"

6. 检查加载后的 metadata
   - 了解文档来源
   - 便于调试和追溯

🎯 使用流程:
1. 选择合适的 Loader（根据文件格式）
2. 配置加载参数（编码、路径等）
3. 调用 load() 或 lazy_load() 加载文档
4. 检查 Document 对象的内容和元数据
5. 进行后续处理（分割、嵌入、存储等）

📚 下一步学习建议:
- 学习文本分割器（Text Splitters）
- 了解向量存储（Vector Stores）
- 探索检索器（Retrievers）的使用
    """)
    
    print("\n✅ 文档加载器学习完成！")


# =============================================================================
# 程序入口点
# =============================================================================

if __name__ == "__main__":
    # 当直接运行此文件时，执行 main() 函数
    # 当作为模块导入时，不会自动执行
    main()
