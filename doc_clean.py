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
RAW_DIR = os.path.join(SOURCE_DIR, "raw")
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
1. 直接输出一个合法的 JSON 对象。
2. **严禁**将 JSON 对象包裹在列表 `[]` 中。
3. **严禁**输出任何 Markdown 格式的标题（如 # 审计报告）或解释性文本。
4. 必须符合以下结构：{{"status": "...", "score": ..., "reason": "..."}}**
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
        # 1. 获取路径信息并生成“唯一安全指纹”
        origin_path = doc.metadata['source']
        origin_dir = os.path.dirname(origin_path)
        display_name = os.path.basename(origin_path)

        # --- 【关键逻辑：路径扁平化】 ---
        # 即使不同文件夹下都有 README.md，rel_path 也会是 "repo1/README.md" 和 "repo2/README.md"
        rel_path = os.path.relpath(origin_path, RAW_DIR)
        # 转换后变成 "repo1_README.md" 和 "repo2_README.md"
        safe_file_name = rel_path.replace(os.sep, "_")

        content = doc.page_content
        content_md5 = calculate_md5(content)

        # 2. 审计与缓存逻辑
        if content_md5 in cache:
            report_data = cache[content_md5]
            report = AuditResult(**report_data)
            print(f"⚡️ 缓存命中: {display_name}")
        else:
            try:
                # 使用 Tavily 搜索辅助审计
                query = f"验证 AI 技术点: {display_name} {content[:100]}"
                search_results = await search_tool.ainvoke(query)

                if len(content) > 15000:
                    chunks = text_splitter.split_text(content)
                    tasks = [chain.ainvoke({'search_results': search_results, 'content': chunk}) for chunk in chunks]
                    all_reports = await asyncio.gather(*tasks)
                    # 聚合逻辑
                    scores = [r.score for r in all_reports]
                    statuses = [r.status for r in all_reports]
                    final_status = "垃圾" if "垃圾" in statuses else max(set(statuses), key=statuses.count)
                    final_score = int(statistics.mean(scores) * 0.4 + min(scores) * 0.6)
                    summary_res = await qwen.ainvoke(
                        f"请简练总结以下审计理由：\n" + "\n".join([r.reason for r in all_reports]))
                    report = AuditResult(status=final_status, score=final_score, reason=summary_res.content)
                else:
                    report = await chain.ainvoke({'search_results': search_results, 'content': content})

                cache[content_md5] = report.model_dump()
            except Exception as e:
                print(f"❌ 审计 {display_name} 失败: {e}")
                return

        # 3. 确定分类文件夹
        if "合格" in report.status:
            target_folder = CLEAN_DIR
        elif "垃圾" in report.status:
            target_folder = TRASH_DIR
        else:
            target_folder = os.path.join(SOURCE_DIR, "Need_Human_Check")
            os.makedirs(target_folder, exist_ok=True)

        # 4. 图片防冲突重处理
        target_pix_dir = os.path.join(target_folder, "picture")
        os.makedirs(target_pix_dir, exist_ok=True)

        img_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
        updated_content = content

        for rel_img_path in img_links:
            if rel_img_path.startswith(('http', 'https')): continue

            # 找到图片的绝对源路径
            abs_img_src = os.path.abspath(os.path.join(origin_dir, rel_img_path))

            if os.path.exists(abs_img_src):
                # --- 【图片重命名逻辑】 ---
                # 图片名同样带上 safe_file_name 前缀，防止不同文档引用同名图片导致覆盖
                raw_img_name = os.path.basename(abs_img_src)
                unique_img_name = f"{os.path.splitext(safe_file_name)[0]}_{raw_img_name}"
                abs_img_dest = os.path.join(target_pix_dir, unique_img_name)

                try:
                    shutil.copy2(abs_img_src, abs_img_dest)
                    # 更新 Markdown 中的引用路径为扁平化后的新路径
                    new_rel_link = f"picture/{unique_img_name}"
                    updated_content = updated_content.replace(rel_img_path, new_rel_link)
                except Exception as e:
                    print(f"⚠️ 图片 {raw_img_name} 迁移失败: {e}")

        # 5. 执行“盖章”与唯一命名移动
        # 最终目的地：target_folder / safe_file_name
        dest = os.path.join(target_folder, safe_file_name)

        try:
            # 写回更新了图片引用路径的内容
            with open(origin_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            # 如果合格，执行 YAML 盖章
            if "合格" in report.status:
                save_metadata_to_file(origin_path, report)

            # 执行物理移动 (Move)
            shutil.move(origin_path, dest)
        except Exception as e:
            print(f"🚚 移动文件 {display_name} 失败: {e}")

        # 6. 播报进度
        print(f"\n✅ 任务完成: {display_name}")
        print(f"🛡️  唯一标识: {safe_file_name}")
        print(f"⚖️  审计结论: {report.status} ({report.score}分)")
        print(f"⚖️  理由: {report.reason})")
        print("-" * 30)


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

    print(f"📂 正在扫描目录: {RAW_DIR}")
    loader = DirectoryLoader(path=RAW_DIR, loader_cls=TextLoader, glob='*.md', recursive=True)
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