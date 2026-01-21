from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader
)

# 显式映射
LOADER_MAPPING = {
    # 文本与代码类
    ".md": (TextLoader, {"encoding": "utf-8"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".py": (TextLoader, {"encoding": "utf-8"}),
    ".json": (TextLoader, {"encoding": "utf-8"}),

    # Word 类
    ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),

    # PPT 类
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),

    # PDF 类
    ".pdf": (PDFMinerLoader, {"concatenate_pages": True}),

    # 表格类
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (TextLoader, {"encoding": "utf-8"}),  # CSV用TextLoader读取更轻量

    # 网页类
    ".html": (UnstructuredHTMLLoader, {}),
    ".htm": (UnstructuredHTMLLoader, {}),
}

import os
import logging
from typing import List
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataIngestion")


def load_industrial_data(data_dir: str) -> List[Document]:
    all_documents = []

    # 1. 扫描磁盘获取所有文件路径 (只扫描一次)
    total_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.startswith('.'):  # 忽略隐藏文件如 .DS_Store
                total_files.append(os.path.join(root, file))

    logger.info(f"📂 发现总文件数: {len(total_files)}")

    # 2. 遍历文件执行路由逻辑
    for file_path in total_files:
        ext = os.path.splitext(file_path)[-1].lower()

        try:
            # --- 场景 A: 命中显式映射 ---
            if ext in LOADER_MAPPING:
                loader_cls, kwargs = LOADER_MAPPING[ext]
                loader = loader_cls(file_path, **kwargs)
                all_documents.extend(loader.load())

            # --- 场景 B: 未命中但可能是文本 (降级捕获) ---
            else:
                # 工业化降级逻辑：尝试用 TextLoader 读取未知后缀
                # 比如 .log, .ini, .cfg 等
                logger.warning(f"⚠️ 未知后缀 {ext}，尝试进入文本降级捕获: {os.path.basename(file_path)}")
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    all_documents.extend(loader.load())
                except Exception:
                    # 如果 TextLoader 也报错，说明是二进制文件 (zip, exe, png等)，彻底跳过
                    logger.error(f"❌ 拒绝解析: {ext} 为不支持的二进制格式或编码错误。")

        except Exception as e:
            logger.error(f"❌ 解析失败 {file_path}: {str(e)}")
            continue

    logger.info(f"✅ 加载完成，总 Documents 块数: {len(all_documents)}")
    return all_documents