import os
import re
from typing import List, Generator
from multiprocessing import Pool, cpu_count

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 配置区 ---
SOURCE_DIR = "/Volumes/soler/PycharmProject2/data/clean"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", " ", ""]
)


def is_clean_text(text: str, threshold: float = 0.85) -> bool:
    """
    清洗算子：增加了对数学公式和 LaTeX 符号的白名单保护。
    """
    if not text.strip():
        return False

    # 1. 检查是否存在 LaTeX 公式特征
    # 包括 $...$, $$, 以及常用的数学命令如 \sum, \alpha, \frac 等
    formula_patterns = [
        r"\$.*?\$",  # 行内公式
        r"\\begin\{.*?\}",  # 环境开始
        r"\\[a-zA-Z]+",  # 反斜杠开头的 LaTeX 命令
        r"[\u2200-\u22FF]",  # Unicode 数学运算符区
        r"[\u0370-\u03FF]"  # Unicode 希腊字母区
    ]

    is_formula_dense = any(re.search(pattern, text) for pattern in formula_patterns)

    # 2. 计算可打印字符占比
    # 增加对常见数学符号的直接放行
    printable_count = sum(1 for char in text if char.isprintable() or char in "\n\t")
    rate = printable_count / len(text)

    # 3. 动态阈值策略
    # 如果检测到是公式密集区，将阈值下调（例如 0.6），因为公式确实包含很多特殊非 ASCII 字符
    effective_threshold = 0.6 if is_formula_dense else threshold

    return rate > effective_threshold


def my_parser(doc: Document) -> List[Document]:
    """
    解析逻辑：输入 1 个完整文档，输出 N 个经过清洗且带元数据的切片文档
    """
    content = doc.page_content
    base_metadata = doc.metadata.copy()

    # 1. 提取审计元数据
    score_match = re.search(r"audit_score: (\d+)", content)
    base_metadata["audit_score"] = int(score_match.group(1)) if score_match else 0

    # 2. 执行切片
    chunks = text_splitter.split_text(content)

    chunked_docs = []
    for i, chunk in enumerate(chunks):
        # 3. 乱码清洗检测
        if not is_clean_text(chunk):
            continue  # 丢弃乱码严重的切片

        # 4. 提取图片路径
        imgs = re.findall(r"!\[.*?\]\((picture/.*?)\)", chunk)

        new_doc = Document(
            page_content=chunk,
            metadata={
                **base_metadata,
                "chunk_id": i,
                "images": imgs,
                "processed_time": "2024-xx-xx"  # 可加入处理时间
            }
        )
        chunked_docs.append(new_doc)
    return chunked_docs


def my_loader(source_dir: str) -> Generator[Document, None, None]:
    """
    装载逻辑：使用 lazy_load 实现流式处理，防止 OOM。
    """
    if not os.path.exists(source_dir):
        print(f"⚠️ 目录不存在: {source_dir}")
        return

    loader = DirectoryLoader(path=source_dir, loader_cls=TextLoader, glob='*.md', recursive=False)
    for raw_doc in loader.lazy_load():
        chunked_list = my_parser(raw_doc)
        for chunk_doc in chunked_list:
            yield chunk_doc


def parallel_process_files(file_paths: List[str]):
    """
    多进程并行
    """
    # 实际应用中会配合 multiprocessing.Pool 使用
    pass
