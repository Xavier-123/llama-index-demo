from typing import List
import qdrant_client
import re

from llama_index.core.llms.llm import LLM

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse

from demo.config.configs import cfg
from demo.custom.template import QA_TEMPLATE, RW_TEMPLATE, OW_TEMPLATE

def queryGenerations(query_str, llm, num_queries=1):
    fmt_rw_prompt = PromptTemplate(RW_TEMPLATE)
    fmt_ow_prompt = PromptTemplate(OW_TEMPLATE)
    rw_response = llm.predict(fmt_rw_prompt, num_queries=num_queries, query_str=query_str)
    ow_response = llm.predict(fmt_ow_prompt, num_queries=num_queries, query_str=query_str)
    # assume LLM proper put each query on a newline
    rw_queries = [re.sub('^\d\.', '', i) for i in rw_response.split("\n")[-2:]]
    ow_queries = [re.sub('^\d\.', '', i) for i in ow_response.split("\n")[-2:]]

    return rw_queries, ow_queries


def merge_node(node_with_scores, node_with_scores_add):
    all_text = [node.text for node in node_with_scores]
    for node in node_with_scores_add:
        if node.text not in all_text:
            node_with_scores.append(node)
    return node_with_scores

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

# import llama_index.core.settings as settings
from llama_index.core import Settings

async def generation_with_knowledge_retrieval(
    query_str: str,
    retriever: BaseRetriever,
    llm: LLM,
    qa_template: str = QA_TEMPLATE,
    reranker: BaseNodePostprocessor | None = None,
    debug: bool = False,
    progress=None,
    embeding_list: list = [],
    settings=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)

    node_with_scores = await retriever.aretrieve(query_bundle)

    settings.embed_model = embeding_list[1][0]
    node_with_scores_embedding_small = await embeding_list[1][1].aretrieve(query_bundle)

    # 合并两个embedding模型检索结果
    all_text = [node.text for node in node_with_scores]
    for node in node_with_scores_embedding_small:
        if node.text not in all_text:
            node_with_scores.append(node)

    # 重写query
    query_split, query_rewrite = [], []
    if cfg["QUERY_REWRITE"]:
        # 拆分
        query_split, query_rewrite = queryGenerations(query_str, llm)
        print("queryGenerations query_split:", query_split)
        print("queryGenerations query_rewrite:", query_rewrite)
        query_split_1_bundle = QueryBundle(query_str=query_split[0])
        query_split_2_bundle = QueryBundle(query_str=query_split[1])
        node_with_scores_query_split_1 = await retriever.aretrieve(query_split_1_bundle)
        node_with_scores_query_split_2 = await retriever.aretrieve(query_split_2_bundle)
        node_with_scores = merge_node(node_with_scores, node_with_scores_query_split_1)
        node_with_scores = merge_node(node_with_scores, node_with_scores_query_split_2)

        # 重写
        for query in query_rewrite:
            if len(query) > 0:
                query_rewrite = query
                break
        query_rewrite_bundle = QueryBundle(query_str=query_rewrite)
        node_with_scores_query_rewrite = await retriever.aretrieve(query_rewrite_bundle)
        node_with_scores = merge_node(node_with_scores, node_with_scores_query_rewrite)


    # 长上下文阅读器
    if cfg["LREORDER"]:
        LCreorder = LongContextReorder()
        node_with_scores = LCreorder.postprocess_nodes(node_with_scores)


    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)
        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    print("llm----->:")
    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret

