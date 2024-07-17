import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from pipeline.embedding import build_embeding
from config.configs import cfg

import time


async def main():
    t1 = time.time()
    # config = dotenv_values(".env")
    config = cfg
    print(config)

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        # model="glm-4",
        model="glm-4-0520",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )

    # embeding
    embeding, retriever = await build_embeding(model_path="F:\inspur\EMBEDDING_MODEL\m3e-base", vector_size=768)
    Settings.embed_model = embeding

    # embeding = HuggingFaceEmbedding(
    #     model_name="F:\inspur\EMBEDDING_MODEL\m3e-base",
    #     # cache_folder="./",
    #     # max_length=768,
    #     embed_batch_size=128,
    # )
    # Settings.embed_model = embeding
    #
    # # 初始化 数据ingestion pipeline 和 vector store
    # client, vector_store = await build_vector_store(config, reindex=False, path="./vector")   # True
    #
    # collection_info = await client.get_collection(
    #     config["COLLECTION_NAME"] or "aiops24"
    # )
    #
    # if collection_info.points_count == 0:
    #     # data = read_data("data")
    #     print("data_dir:", cfg["DATA_DIR"])
    #     data = read_data(cfg["DATA_DIR"])
    #     pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
    #     # 暂时停止实时索引
    #     await client.update_collection(
    #         collection_name=config["COLLECTION_NAME"] or "aiops24",
    #         optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
    #     )
    #     await pipeline.arun(documents=data, show_progress=True, num_workers=1)
    #     # 恢复实时索引
    #     await client.update_collection(
    #         collection_name=config["COLLECTION_NAME"] or "aiops24",
    #         optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    #     )
    #     print(len(data))
    #
    # retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=3)

    # queries = read_jsonl("question-md.jsonl")
    queries = read_jsonl("question-pdd.jsonl")

    # 生成答案
    print("Start generating answers...")

    t2 = time.time()
    results = []
    for query in tqdm(queries, total=len(queries)):
        # # query["query"] = "怎么查看二代卡？"
        # query["query"] = "怎么生成RANK_TABLE_FILE的json文件？"
        result = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm
        )
        results.append(result)
        print(result)
        print("-"*50)
        # break

    t3 = time.time()
    print("time t2 - t1:", t2 - t1)
    print("time t3 - t2:", t3 - t2)

if __name__ == "__main__":
    asyncio.run(main())
