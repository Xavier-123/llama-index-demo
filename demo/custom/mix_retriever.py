import asyncio
from tqdm.asyncio import tqdm
from llama_index.core import Settings
from llama_index.legacy.llms import OpenAILike

def build_llm():
    from llama_index.legacy.llms import OpenAILike
    llm = OpenAILike(
        api_key='4674c23251e02b9c04b6d2c78e73a7a3.5azT9pxx0cHPX89Q',
        model="glm-4",
        # model="glm-4-0520",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
        # max_tokens=200
    )
    # Settings.llm = llm
    return llm


def build_embedd():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embeding = HuggingFaceEmbedding(
        model_name=r'E:\work\bge-small-zh-v1.5',  # 512
        # cache_folder="./",
        embed_batch_size=256,
        max_length=512,
        query_instruction='为这个句子生成表示以用于检索相关文章：'
    )
    # Settings.embeding = embeding
    return embeding


def read_document():
    from llama_index.core import SimpleDirectoryReader
    documents = SimpleDirectoryReader(
        input_dir="../data/md",
        recursive=True,
        required_exts=[
            ".txt",
            ".md",
            ".docx",
            ".pdf",
            ".xlsx",
        ],
    ).load_data()
    return documents

def build_node_index(documents, embedding):
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    # 初始化存储上下文（默认为内存中）
    storage_context = StorageContext.from_defaults()

    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embedding)
    return nodes, vector_index

async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    _queries = [queries[0] for _ in range(len(task_results))]
    # for i, (query, query_result) in enumerate(zip(queries, task_results)):
    for i, (query, query_result) in enumerate(zip(_queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict

async def bm25():
    from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core import Settings


    llm = build_llm()
    Settings.llm = llm
    embedding = build_embedd()
    Settings.embed_model = embedding
    documents = read_document()

    # Todo 构建node和索引
    nodes, vector_index = build_node_index(documents, Settings.embed_model)

    # 向量检索器
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)

    # bm25关键词检索器
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=vector_index.docstore, similarity_top_k=2
    )
    mix_retriever_response = await run_queries(
        ["查看芯片是一代还是二代"],
        [bm25_retriever, vector_retriever]
    )
    print("mix_retriever_response:", len(mix_retriever_response), mix_retriever_response)




if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bm25())