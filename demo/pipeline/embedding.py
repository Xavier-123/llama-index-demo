from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from demo.pipeline.ingestion import build_pipeline, build_vector_store, read_data
from demo.pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from demo.config.configs import cfg
from qdrant_client import models
import json

async def build_embedding_retriever(
        model_path,
        # vector_size
):
    global embedding
    embedding = HuggingFaceEmbedding(
                model_name=model_path,
                # cache_folder="./",
                embed_batch_size=256,
                # max_length =768 ,
                # encode_kwargs = {'normalize_embeddings': True},
                query_instruction='为这个句子生成表示以用于检索相关文章：'
            )

    with open(model_path + "/config.json", 'r') as f:
        model_config = json.loads(f.read())

    model_name = model_path.replace("\\", "/").split("/")[-1]
    cfg["VECTOR_SIZE"] = model_config["hidden_size"]

    print(f'Init pipeline and vector store {model_name}')
    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(cfg, storepath=f'./vector/{model_name}/', reindex=cfg["REINDEX"])   # reindex=True 重制作向量

    collection_info = await client.get_collection(
        cfg["COLLECTION_NAME"] or "aiops24"
    )
    print(collection_info.points_count)

    if collection_info.points_count == 0:
        data = read_data(cfg["DATA_DIR"] or "data")
        pipeline = build_pipeline(embed_model=embedding,
                                  vector_store=vector_store,
                                  # themetree=treedict
                                  )
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=cfg["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=cfg["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(f"data length: {len(data)}")
        # Update collection info !
        collection_info = await client.get_collection(
            cfg["COLLECTION_NAME"] or "aiops24"
        )
        print(f"points count:{collection_info.points_count}")

    retriever = QdrantRetriever(vector_store, embedding, similarity_top_k=10)

    return embedding, retriever, cfg["VECTOR_SIZE"]