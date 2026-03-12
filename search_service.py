import numpy as np
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

MODEL_PATH = "/Volumes/soler/PycharmProject2/models/bge-m3"
COLLECTION_NAME = "audit_bge_m3_collection"
MILVUS_URI = "http://localhost:19530"

# 初始化全局连接和模型
client = MilvusClient(uri=MILVUS_URI)
embeddings = BGEM3EmbeddingFunction(model_name=MODEL_PATH, device='mps')


def search_with_quality_gate(query_str: str, top_k: int = 3, min_audit_score: float = 75.0):
    """
    双轨制检索：RRF选拔 + Dense Score 质量锚定 + Metadata 评分过滤
    :param query_str: 用户查询
    :param top_k: 返回数量
    :param min_audit_score: 业务评分阈值，低于此分数的元数据将被舍弃
    :return: 包含绝对相似度分值的检索结果列表
    """
    # 1. 加载集合
    client.load_collection(COLLECTION_NAME)
    query_output = embeddings.encode_queries([query_str])

    # 2. 构造检索请求
    # 💡 稠密向量路 (Dense)：使用 IP (内积)，在归一化后等同于余弦相似度
    search_params_dense = {"metric_type": "IP", "params": {"ef": 64}}
    req_dense = AnnSearchRequest(
        data=query_output["dense"],
        anns_field="dense_vector",
        param=search_params_dense,
        limit=top_k * 5
    )

    # 3. 稀疏向量路 (Sparse)
    search_params_sparse = {"metric_type": "IP"}
    req_sparse = AnnSearchRequest(
        data=query_output["sparse"],
        anns_field="sparse_vector",
        param=search_params_sparse,
        limit=top_k * 5
    )

    # 4. 执行混合检索 (RRF 排名)
    # 注意：hybrid_search 返回的结果中，'distance' 是 RRF 分数
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req_dense, req_sparse],
        ranker=RRFRanker(k=60),
        limit=top_k * 2,
        output_fields=["content", "metadata", "dense_vector"]
    )

    hits = res[0]
    final_results = []

    # 5. 计算绝对相似度 (Quality Gate)
    # 因为 RRF 丢失了原始物理距离，我们需要拿到 Top 1 之后返回它在 Dense 维度的原始得分作为 Agent 反思的依据。
    q_vec = np.array(query_output["dense"][0])
    for hit in hits:
        metadata = hit['entity'].get('metadata', {})
        data_score = metadata.get('score', 0)
        if data_score < min_audit_score:
            continue
        result_item = {
            "id": hit['id'],
            "rrf_score": round(hit['distance'], 4),  # 相对排名分
            "abs_threshold": None,  # 待填充的绝对相似度
            "content": hit['entity'].get('content'),
            "window_context": metadata.get('window_context', "无背景"),
            "metadata": metadata
        }

        doc_vec = np.array(hit['entity'].get('dense_vector', []))
        if doc_vec.size > 0:
            # 计算余弦相似度作为“质量门禁”的指标
            abs_score = np.dot(q_vec, doc_vec)
            result_item["abs_threshold"] = round(float(abs_score), 4)

        final_results.append(result_item)
        # 设置结束条件
        if len(final_results) >= top_k:
            break

    return final_results
