from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.postprocessor import SentenceTransformerRerank

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from pipeline.embedding import build_embedding_retriever
from config.configs import cfg
from qdrant_client import models
from tqdm.asyncio import tqdm
import time
import asyncio
from dotenv import dotenv_values


async def main():
    t1 = time.time()
    # config = dotenv_values(".env")
    config = cfg
    print(config)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager

    # 初始化 LLM 嵌入模型
    llm = OpenAILike(
        api_key=config["GLM_KEY"],
        model="glm-4",
        # model="glm-4-0520",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
        # max_tokens=200
    )

    # llm = OpenAILike(
    #     api_key='fake',
    #     model="qwen2-72b-instruct-int4",
    #     api_base="http://10.108.1.254:18001/v1",
    #     max_tokens=512
    # )

    # embeding
    embeding_retriever_list = []
    embeding, retriever, vector_store_index = await build_embedding_retriever(
        model_path="F:\inspur\EMBEDDING_MODEL\m3e-base",
        # vector_size=768
    )
    Settings.embed_model = embeding
    embeding_retriever_list.append([embeding, retriever, vector_store_index, 768])

    embeding_small, retriever_small, vector_store_index_small = await build_embedding_retriever(
        model_path="F:\inspur\EMBEDDING_MODEL\m3e-small",
        # vector_size=512
    )
    embeding_retriever_list.append([embeding_small, retriever_small, vector_store_index, 512])


    # build index
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core import StorageContext

    documents = read_data(cfg["DATA_DIR"])
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)

    # 初始化存储上下文（默认为内存中）
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # 定义向量索引和关键词表索引
    from llama_index.core import GPTVectorStoreIndex, SimpleKeywordTableIndex, VectorStoreIndex

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    vector_index.docstore.persist(persist_path="./storage/vector_index")
    keyword_index.docstore.persist(persist_path="./storage/keyword_index")


    # reranker
    reranker = None
    if cfg["RERANKER_MODEL"]:
        # Re-Rank the top 3 chunks based on the gpt-3.5-turbo-0125 model
        reranker = SentenceTransformerRerank(model=r"F:\inspur\EMBEDDING_MODEL\Xorbits\bge-reranker-base", top_n=6)

    queries = read_jsonl("question-pdd.jsonl")

    # 生成答案
    print("Start generating answers...")

    t2 = time.time()
    results = []
    for query in tqdm(queries, total=len(queries)):
        # # query["query"] = "怎么查看二代卡？"
        # query["query"] = "怎么生成RANK_TABLE_FILE的json文件？"
        query["query"] = "如何部署一个AI平台？"
        result = await generation_with_knowledge_retrieval(
            query_str=query["query"],
            # retriever=retriever,
            llm=llm,
            reranker=reranker,
            embeding_retriever_list=embeding_retriever_list,
            debug=cfg["DEBUG"],
            settings=Settings
        )
        results.append(result)
        print(result)
        print("-" * 50)
        break

    t3 = time.time()
    print("time t2 - t1:", t2 - t1)
    print("per response:", (t3 - t2) / 2)


if __name__ == "__main__":
    asyncio.run(main())
