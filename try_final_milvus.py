import os
import time
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from ingest_to_milvus import load_data
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
SOURCE_DIR = "/Volumes/soler/PycharmProject2/data/clean"
MODEL_PATH = "/Volumes/soler/PycharmProject2/models/bge-m3"
COLLECTION_NAME = "audit_bge_m3_collection"
BATCH_SIZE = 8

# 初始化本地embedding模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'

 #连接milvus
print("🔌 正在连接 Milvus...")
connections.connect("default", host="localhost", port="19530")
embeddings = BGEM3EmbeddingFunction(
    model_name=MODEL_PATH,
    device='mps',
    use_fp16=False
    )

def setup_collection():
    DIMENSION = embeddings.dim["dense"]
    print(f"✅ 模型加载成功！稠密向量维度: {DIMENSION}")
    #搭建milvus的scheme
    field = [
        #主键
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        #Sparse向量
        FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
        #Dense向量
        FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
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
    print("📊 正在配置双路索引...")

    # 1. 稠密向量索引 (HNSW)
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 256}
    }
    collection.create_index(field_name="dense_vector", index_params=dense_index)

    # 2. 稀疏向量索引 (SPARSE_INVERTED_INDEX)
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
    collection.create_index(field_name="sparse_vector", index_params=sparse_index)
    return collection

def insert_to_milvus(collection, model, docs):
    # 1. 批量提取文本并生成向量
    contents = [d.page_content for d in docs]
    output = model.encode_documents(contents)
    dense_vecs = output["dense"]
    sparse_vecs = output["sparse"]

    formatted_sparse_vecs = []
    # raw_sparse_vecs 是 scipy 格式更改
    for i in range(sparse_vecs.shape[0]):
        row = sparse_vecs[i].tocoo()
        formatted_sparse_vecs.append(dict(zip(row.col, row.data)))
    # 2. 准备其他标量字段
    scores = [int(d.metadata.get('score', 0)) for d in docs]
    metadatas = [d.metadata for d in docs]

    # 3. 组装插入格式：[向量列表, 文本列表, 分数列表, 元数据列表]
    # 对应 Schema 顺序
    data = [
        formatted_sparse_vecs, # 对应 sparse_vecs 字段
        dense_vecs,  # 对应 dense_vecs 字段
        contents,  # 对应 content 字段
        scores,  # 对应 score 字段
        metadatas  # 对应 metadata 字段
    ]

    collection.insert(data)

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



if __name__ == "__main__":
    main()
