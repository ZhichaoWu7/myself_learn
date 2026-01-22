import asyncio
import hashlib
import json
import os
import statistics
import tempfile
import shutil
import re

from datetime import datetime
from anyio import Semaphore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

search_tool = TavilySearch(api_key=os.getenv('TAILY_API_KEY'),
                           max_results=5)
qwen = ChatOpenAI(model="qwen-max",
                  base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                  api_key=os.getenv('DASHSCOPE_API_KEY'),
                  streaming=False)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
current_date = datetime.now().strftime('%Y-%m-%d')
SOURCE_DIR = "/Volumes/soler/PycharmProject2/data"
CLEAN_DIR = os.path.join(SOURCE_DIR, "clean")
TRASH_DIR = os.path.join(SOURCE_DIR, "trash")
CACHE_FILE = os.path.join(SOURCE_DIR, "cache.json") #新增缓存文件路径
#创建新文件夹
for fold in [CLEAN_DIR, TRASH_DIR]:
    os.makedirs(fold, exist_ok=True)

prompt = ChatPromptTemplate.from_template(
"""
你是一个深度学习技术审计专家。请对文档进行【分类审计】。

[参考信息]：{search_results}
[待审计内容]：{content}

[审计逻辑]：
1. **理论基础类**：若内容为数学推导、算法原理（如 CNN/RNN 原理），只要逻辑严密，评分应在 85+，不强制要求 API 时效。
2. **代码实践类**：若包含具体实现，必须符合当前主流（PyTorch 2.x+, LangChain 0.3+）。使用 Caffe/Theano 或 PyTorch 1.x 旧语法（如 .data, Variable）的，判定为 [存疑] 或 [垃圾]。
3. **一票否决**：严禁内容注水、逻辑谬误。

[评分阶梯]：
- 90-100: [优质] 理论深厚或包含 2024+ 生产级代码。
- 75-89: [合格] 经典理论，或技术栈略旧但思路完全正确。
- 60-74: [存疑] 缺乏代码、排版混乱、或技术栈已完全过时（如 TensorFlow 1.x）。
- 60以下: [垃圾] 严重错误、废弃技术、或无意义内容。

[输出规范]：
直接输出一个 JSON 对象，包含 status, score, reason 字段。
**绝对禁止将对象包裹在列表 [] 中。**
"""
)
class AuditResult(BaseModel):
    status: str = Field(description='判定结果，只能是合格 或 垃圾 或 存疑')
    score: int = Field(description='0-100之间的数字')
    reason: str = Field(description='判定的详细原因')
structured_llm = qwen.with_structured_output(AuditResult)
chain = prompt | structured_llm



#辅助函数
def calculate_md5(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

#长期记忆的缓存读取
def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f: return json.load(f)
    return {}

#缓存保存
def save_cache(cache_file: str, data: dict):
    with open(cache_file, 'w') as f: json.dump(data, f, ensure_ascii=False)


async def processed_doc(doc: Document, semaphore: Semaphore, cache: dict) -> None:
    async with semaphore:
        origin_path = doc.metadata['source']
        file_name = os.path.basename(origin_path)
        origin_dir = os.path.dirname(origin_path)  # 获取原文件所在目录
        content = doc.page_content
        content_md5 = calculate_md5(content)

        # 1. 缓存与审计逻辑 (完全保留你的逻辑)
        if content_md5 in cache:
            report_data = cache[content_md5]
            report = AuditResult(**report_data)
            print(f"⚡️ 缓存命中: {file_name}")
        else:
            try:
                query = f"验证ai技术点: {file_name} {content[:100]}"
                result = await search_tool.ainvoke(query)

                if len(content) > 15000:
                    chunks = text_splitter.split_text(content)
                    tasks = [chain.ainvoke({'search_results': result, 'content': chunk}) for chunk in chunks]
                    all_reports = await asyncio.gather(*tasks)

                    scores = [r.score for r in all_reports]
                    statuses = [r.status for r in all_reports]
                    final_status = "垃圾" if "垃圾" in statuses else max(set(statuses), key=statuses.count)
                    final_score = int(statistics.mean(scores) * 0.4 + min(scores) * 0.6)

                    all_reasons_text = '\n'.join([f'块{i + 1}: {r.reason}' for i, r in enumerate(all_reports)])
                    summary_res = await qwen.ainvoke(
                        f"你是一个总结专家，请用一段话精炼总结以下审计理由，直接输出结论：\n{all_reasons_text}")
                    report = AuditResult(status=final_status, score=final_score, reason=summary_res.content)
                else:
                    report = await chain.ainvoke({'search_results': result, 'content': content})

                cache[content_md5] = report.model_dump()
            except Exception as e:
                print(f"❌ 处理 {file_name} 时出错: {e}")
                return

        # 2. 确定分类文件夹并准备图片文件夹
        target_folder = CLEAN_DIR if "合格" in report.status else (
            TRASH_DIR if "垃圾" in report.status else os.path.join(SOURCE_DIR, "Need_Human_Check"))

        # 新增：在分类目录下创建 picture 文件夹
        target_pix_dir = os.path.join(target_folder, "picture")
        os.makedirs(target_pix_dir, exist_ok=True)

        # 新增：同步搬运图片文件
        img_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
        for rel_img_path in img_links:
            if rel_img_path.startswith(('http', 'https')): continue

            # 物理路径溯源：根据 md 里的路径找到硬盘里的图
            abs_img_src = os.path.abspath(os.path.join(origin_dir, rel_img_path))

            if os.path.exists(abs_img_src):
                img_name = os.path.basename(abs_img_src)
                abs_img_dest = os.path.join(target_pix_dir, img_name)
                try:
                    # 用 copy2 保证图片属性不变，且不破坏其他 md 对原图的引用
                    shutil.copy2(abs_img_src, abs_img_dest)
                except Exception as e:
                    print(f"⚠️ 图片搬运失败: {img_name} | {e}")

        # 3. 执行移动与盖章 (保留你的原有逻辑)
        dest = os.path.join(target_folder, file_name)

        # 如果是合格，先打戳再移动
        if "合格" in report.status:
            save_metadata_to_file(origin_path, report)

        # 移动 MD 文件
        shutil.move(origin_path, dest)

        # 4. 实时播报 (完全按照你的要求保留)
        print("\n" + "=" * 50)
        print(f"✅ 处理完成 >> 📄 {file_name}")
        print(f"⚖️ 结论: {report.status} | 分数: {report.score}")
        print(f"📝 理由: {report.reason}")
        print(f"🖼️ 关联图片已同步至: {target_pix_dir}")
        print("=" * 50)


def save_metadata_to_file(filename: str, metadata: AuditResult) -> None:
    # 在同一目录下创建临时文件
    target_dir = os.path.dirname(filename)
    fd, temp_path = tempfile.mkstemp(dir=target_dir, text=True)
    try:
        # 使用 os.fdopen 确保 fd 正确关闭
        with os.fdopen(fd, "w", encoding='utf-8') as temp_file:
            safe_reason = metadata.reason.replace('\n', ' ').replace('"', '\\"')
            temp_file.write("---\n")
            temp_file.write(f"audit_status: \"{metadata.status}\"\n")
            temp_file.write(f"audit_score: {metadata.score}\n")
            temp_file.write(f"audit_reason: \"{safe_reason}\"\n")
            temp_file.write(f"audit_date: \"{current_date}\"\n")
            temp_file.write("---\n\n")

            with open(filename, "r", encoding='utf-8') as old_file:
                shutil.copyfileobj(old_file, temp_file)
        shutil.move(temp_path, filename)
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"写入元数据失败: {e}")


async def main():
    # 启动载入缓存
    cache = load_cache(CACHE_FILE)

    print(f"📂 正在扫描目录: {SOURCE_DIR}")
    loader = DirectoryLoader(path=SOURCE_DIR, loader_cls=TextLoader, glob='*.md', recursive=False)
    docs = loader.load()

    semaphore = asyncio.Semaphore(2)

    # 将 cache 字典传入
    tasks = [processed_doc(doc, semaphore, cache) for doc in docs]
    await asyncio.gather(*tasks)

    # 运行结束后保存一次缓存
    save_cache(CACHE_FILE, cache)
    print("🎉 异步审计及物理分类全部完成！")



if __name__ == '__main__':
    asyncio.run(main())