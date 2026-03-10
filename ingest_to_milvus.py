import os
import re
import datetime
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
    if not text.strip():
        return False

    formula_patterns = [
        r"\$.*?\$",  # 行内公式
        r"\\begin\{.*?\}",  # 环境开始
        r"\\[a-zA-Z]+",  # 反斜杠开头的 LaTeX 命令
        r"[\u2200-\u22FF]",  # Unicode 数学运算符区
        r"[\u0370-\u03FF]"  # Unicode 希腊字母区
    ]

    is_formula_dense = any(re.search(pattern, text) for pattern in formula_patterns)

    printable_count = sum(1 for char in text if char.isprintable() or char in '\n\t')
    rate = printable_count / len(text)
    effctive_threshold = 0.6 if rate > threshold else threshold
    return rate > effctive_threshold

def my_parser(doc: Document, window_size: int = 3) -> List[Document]:
    now_date = datetime.datetime.now().strftime("%Y%m%d")
    chunk_list = []
    content = doc.page_content
    base_metadata = doc.metadata.copy()

    #匹配元数组
    raw_source = base_metadata.get('source', 'unknown')
    file_name = os.path.basename(raw_source)
    score_match = re.search(r"audit_score: (\d+)", content)
    reason_match = re.search(r"audit_reason: \s*(.*)", content)
    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1) if reason_match else ""

    chunks = text_splitter.split_text(content)
    for i, chunk in enumerate(chunks):
        if not is_clean_text(chunk):
            continue
        start = max(0, i - window_size)
        end = min(i + window_size + 1, len(chunks))
        full_window = "".join(chunks[start:end])
        enhanced_content = f"【审计判定：{reason}】\n原始内容：{chunk}"
        img = re.findall(r"!\[.*?\]\((picture/.*?)\)", enhanced_content)
        new_doc = Document(
            page_content = enhanced_content,
            metadata = {
                'window_context': full_window,
                'file_name': file_name,
                'chunk_id': i,
                'score': score,
                "images": img,
                "processed_time": now_date
            }
        )
        chunk_list.append(new_doc)
    return chunk_list


def load_data(source_dir: str) -> Generator[Document, None, None]:
    if not os.path.exists(source_dir):
        print(f"⚠️ 目录不存在: {source_dir}")
        return

    loader = DirectoryLoader(source_dir, loader_cls=TextLoader, glob='*.md', recursive=False)
    for doc in loader.lazy_load():
        chunk_list = my_parser(doc)
        for chunk in chunk_list:
            yield chunk


