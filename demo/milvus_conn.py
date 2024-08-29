from qdrant_client import AsyncQdrantClient, models
#
# client = AsyncQdrantClient(
#     # url=config["QDRANT_URL"],
#     # location=":memory:",
#     path="./123/"
# )


def llama_index_milvus_test():
    # Create an index over the documents
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.milvus import MilvusVectorStore


    vector_store = MilvusVectorStore(
        uri="http://192.168.12.167:19530", dim=512, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        "./data/md/", storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    res = query_engine.query("What did the author learn?")
    print(res)


def pymilvus_test():
    import random
    from pymilvus import MilvusClient
    def generate_random_vector(dim):
        """生成随机向量"""
        return [random.random() for _ in range(dim)]

    HOST = "10.108.1.254"
    PORT = 18090
    URI = f"http://{HOST}:{PORT}/fastgpt"
    client = MilvusClient(
        uri=URI,
        token="root:My_#741"
    )
    metric_type = 'COSINE'
    VECTOR_DIMENSION = 1536
    search_vector = [generate_random_vector(VECTOR_DIMENSION)]  # 每次请求前生成随机向量

    search_params = {
        "metric_type": metric_type,
        "params": {'M': 64, 'efConstruction': 32}
    }
    COLLECTION_NAME = 'modeldata'
    TOPK = 1

    # todo 查询
    try:
        response = client.search(
            collection_name=COLLECTION_NAME,
            data=search_vector,
            limit=TOPK,
            search_params=search_params
        )
        if response:
            extra_info = response.extra
            extra_info_cost = extra_info.get('cost', 0)  # 获取额外信息中的成本
    except:
        pass

    # todo 新增
    try:
        client.insert()
    except:
        pass

    # todo 修改
    try:
        client.upsert()
    except:
        pass

    # todo 删除
    try:
        client.delete()
    except:
        pass



if __name__ == '__main__':
    pymilvus_test()

