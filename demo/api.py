import uvicorn
from fastapi import FastAPI, Form, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from llama_index.core import Settings
from llama_index.legacy.llms import OpenAILike
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from pipeline.embedding import build_embedding_retriever
from config.configs import cfg
from tqdm.asyncio import tqdm
import time
import asyncio
from dotenv import dotenv_values
from pydantic import BaseModel, Field

app = FastAPI()


class RequestModel(BaseModel):
    user_id: str = Field("123", example="123")
    knowledge_id: str = Field("321", example="321")
    query: str = Field("怎么查询数据专线业务数据?", example="")
    api_key: str = Field("", example="fake")
    api_base: str = Field("", example="")
    model_name: str = Field("", example="")
    max_tokens: int = Field(512, example="")
    reranker_top: int = Field(7, example="", description="获取重排序的top节点数据")


class Embed_Retriever():
    embedding_small: HuggingFaceEmbedding = None
    embedding_base: HuggingFaceEmbedding = None
    embedding_large: HuggingFaceEmbedding = None
    retriever: BaseRetriever = None
    key_retriever: BaseRetriever = None
    reranker_retriever: BaseRetriever = None


llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# _llm = OpenAILike(
#     api_key='fake',
#     model="qwen2-72b-instruct-int4",
#     api_base="http://10.108.1.254:18001/v1",
#     # api_key=cfg["GLM_KEY"],
#     # model="glm-4",
#     # api_base="https://open.bigmodel.cn/api/paas/v4/",
#     # is_chat_model=True,
#     max_tokens=512
# )

embeding_retriever_list = []
reranker_list = []
bm25_list = []


async def api_build_embedding(embeding_retriever_list: list, reranker_list: list, bm25_list: list):
    for embedding_path in cfg["EMBEDDING_LIST"]:
        embeding, retriever, vector_size = await build_embedding_retriever(
            model_path=embedding_path,
            # vector_size=768
        )
        Settings.embed_model = embeding
        embeding_retriever_list.append([embeding, retriever, vector_size])

    # if len(cfg["EMBEDDING_LIST"]) > 1:
    #     embeding_small, retriever_small = await build_embedding_retriever(
    #         model_path=cfg["EMBEDDING_LIST"][1],
    #         # vector_size=512
    #     )
    #     embeding_retriever_list.append([embeding_small, retriever_small, 512])

    if cfg["RERANKER_MODEL"]:
        # Re-Rank the top 3 chunks based on the gpt-3.5-turbo-0125 model
        reranker_list.append(SentenceTransformerRerank(model=cfg["RERANKER_MODEL"], top_n=10))

    if cfg["MIXED_SEARCH"]:
        from llama_index.retrievers.bm25 import BM25Retriever
        from demo.pipeline.ingestion import read_data
        from demo.custom.mix_retriever import build_node_index

        documents = read_data(cfg["DATA_DIR"])
        nodes, vector_index = build_node_index(documents, Settings.embed_model)
        # bm25关键词检索器
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=vector_index.docstore, similarity_top_k=2
        )
        bm25_list.append(bm25_retriever)

    return embeding_retriever_list, reranker_list, bm25_list


@app.post("/chaxun")
async def chaxun(
        req: RequestModel,
):
    t_start = time.time()

    _llm = OpenAILike(
        api_key='fake',
        model="qwen2-72b-instruct-int4",
        api_base="http://10.108.1.254:18003/v1",
        is_chat_model=True,
        # max_tokens=256
    )
    # Settings.llm = _llm

    llm = _llm
    # if req.api_base is not None and req.model_name is not None and len(req.model_name) > 0 and len(req.api_base) > 0:
    #     llm = OpenAILike(
    #         api_key=req.api_key,
    #         model=req.model_name,
    #         api_base=req.api_base,
    #         max_tokens=512
    #     )

    # result = await generation_with_knowledge_retrieval(
    result, quote = await generation_with_knowledge_retrieval(
        query_str=req.query,
        # retriever=retriever,
        # bm25=bm25,
        llm=llm,
        reranker=reranker,
        reranker_top=req.reranker_top,
        embeding_retriever_list=embeding_retriever_list,
        debug=cfg["DEBUG"],
        settings=Settings
    )
    t_end = time.time()

    print(result)
    content = {
        "isSuc": True,
        "code": 0,
        "msg": "Success ~",
        "res": {
            "text": result.text,
            "quote": quote,
            "time": str(t_end - t_start),
            "model_name": result.raw["model"]
        }
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


if __name__ == '__main__':
    asyncio.run(api_build_embedding(embeding_retriever_list, reranker_list, bm25_list))
    retriever = embeding_retriever_list[0][1]
    reranker = reranker_list[0]
    if len(bm25_list) > 0:
        bm25 = bm25_list[0]

    uvicorn.run(app, host='127.0.0.1', port=9111)
