import os
import time
import re
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from ingest_to_milvus import load_data
SOURCE_DIR = "/Volumes/soler/PycharmProject2/data/clean"
MODEL_PATH = "/Volumes/soler/PycharmProject2/models/bge-m3"
COLLECTION_NAME = "audit_bge_m3_collection"
BATCH_SIZE = 8

# 初始化本地embedding模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'

 #连接milvus
print("🔌 正在连接 Milvus...")
connections.connect("default", host="localhost", port="19530")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH,
                                           model_kwargs={"device": 'mps'},
                                           encode_kwargs={"normalize_embeddings": True})

def setup_collection():
    try:
        test_text = embeddings.embed_query("测试文本")
        DIMENSION = len(test_text)
        print(f"✅ 模型加载成功！向量维度: {DIMENSION}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)
    #搭建milvus的scheme
    field = [
        #主键
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        #向量
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        #初始文本
        FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=65535),
        #审计分数
        FieldSchema(name='score', dtype=DataType.INT8),
        #元数据
        FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

    schema = CollectionSchema(fields=field, description="审计文档RAG集合")
    print("Schema 结构:")
    print(schema)
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ 集合 {COLLECTION_NAME} 已存在，正在删除重建以演示流程...")
        utility.drop_collection(COLLECTION_NAME)
    print(f"正在创建集合: {COLLECTION_NAME} ...")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("集合创建成功！")

    print("正在配置HNSW索引...")
    index_params = {"metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
             }
    collection.create_index(field_name="vector", index_params=index_params)
    return collection

def main():
    collection = setup_collection()
    batch_docs = []
    total_count = 0
    start_time = time.time()

    print(f"🚀 开始从目录加载数据: {SOURCE_DIR}")
    doc_generator = load_data(SOURCE_DIR)

    # 使用 tqdm 显示进度
    for doc in tqdm(doc_generator, desc="向量化并入库"):
        batch_docs.append(doc)
        if len(batch_docs) >= BATCH_SIZE:
            insert_to_milvus(collection, embeddings, batch_docs)
            total_count += len(batch_docs)
            batch_docs = []

    if batch_docs:
        insert_to_milvus(collection, embeddings, batch_docs)
        total_count += len(batch_docs)

    print("正在执行 Flush 并加载到内存...")
    collection.flush()
    collection.load()

    duration = time.time() - start_time
    print("-" * 30)
    print(f"✅ 入库圆满完成！")
    print(f"📦 总存入切片数: {total_count}")
    print(f"⏱️ 总耗时: {duration:.2f} 秒")
    print(f"📈 集合当前实体总数: {collection.num_entities}")
    print("-" * 30)


def insert_to_milvus(collection, model, docs):
    # 1. 批量提取文本并生成向量
    contents = [d.page_content for d in docs]
    vectors = model.embed_documents(contents)

    # 2. 准备其他标量字段
    scores = [int(d.metadata.get('score', 0)) for d in docs]
    metadatas = [d.metadata for d in docs]

    # 3. 组装插入格式：[向量列表, 文本列表, 分数列表, 元数据列表]
    # 对应 Schema 顺序
    data = [
        vectors,  # 对应 vector 字段
        contents,  # 对应 content 字段
        scores,  # 对应 score 字段
        metadatas  # 对应 metadata 字段
    ]

    collection.insert(data)


if __name__ == "__main__":
    main()
