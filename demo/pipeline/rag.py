from typing import List
import qdrant_client
import re

from llama_index.core.llms.llm import LLM
from llama_index.legacy.llms import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever, __all__
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent
import llama_index.core.instrumentation as instrument

from demo.config.configs import cfg
from demo.custom.template import QA_TEMPLATE, HYDE_TEMPLATE, RW_TEMPLATE, OW_TEMPLATE
from demo.custom.transformation import CustomQueryEngine, CustomSentenceTransformerRerank

dispatcher = instrument.get_dispatcher(__name__)


def hydeGenerations(query_str: str, llm, num_queries=1):
    hyde_prompt = PromptTemplate(HYDE_TEMPLATE)
    hyde_response = llm.predict(hyde_prompt, num_queries=num_queries, query_str=query_str)
    hyde_queries_list = [re.sub('^\d\.', '', i) for i in hyde_response.split("\n")[-2:]]
    return query_str + "#" + hyde_queries_list[0]


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

def mix_retriever(embeding_list):
    from llama_index.core.retrievers import QueryFusionRetriever
    index_1, index_2 = embeding_list[0][2], embeding_list[1][2]
    retriever = QueryFusionRetriever(
        [index_1.as_retriever(), index_2.as_retriever()],
        similarity_top_k=2,
        num_queries=4,  # set this to 1 to disable query generation
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )
    return retriever


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

    async def _aretrieve(self, query_bundle: QueryBundle, qdrant_filters=None) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = await self._vector_store.aquery(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    @dispatcher.span
    async def aretrieve_cc(self, str_or_query_bundle: QueryType, qdrant_filters=None) -> List[NodeWithScore]:
        self._check_callback_manager()
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                    CBEventType.RETRIEVE,
                    payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle=query_bundle, qdrant_filters=qdrant_filters)
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

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


async def generation_with_knowledge_retrieval(
        query_str: str,
        retriever: BaseRetriever,
        # retriever: QdrantRetriever,
        # llm: LLM,
        llm: OpenAILike,
        qa_template: str = QA_TEMPLATE,
        reranker: BaseNodePostprocessor | None = None,
        reranker_top: int | None = None,
        debug: bool = False,
        progress=None,
        embeding_list: list = [],
        settings=None,
) -> CompletionResponse:
    query_bundle = QueryBundle(query_str=query_str)

    node_with_scores = await retriever.aretrieve(query_bundle)

    if len(embeding_list) > 1:
        settings.embed_model = embeding_list[1][0]
        node_with_scores_embedding_small = await embeding_list[1][1].aretrieve(query_bundle)

        if debug:
            print(f"node_with_scores_embedding_small:\n{node_with_scores_embedding_small}\n------")

        settings.embed_model = embeding_list[0][0]
        # 合并两个embedding模型检索结果-直接将前3个node合并到node_with_scores
        node_with_scores.extend(node_with_scores_embedding_small[:3])

        # all_text = [node.text for node in node_with_scores]
        # for node in node_with_scores_embedding_small:
        #     if node.text not in all_text:
        #         node_with_scores.append(node)

    _mix_retriever = mix_retriever(embeding_list)
    _mix_retriever.retrieve()



    # 重写query
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

        if debug:
            print(f"query_rewrite_bundle:\n{query_rewrite_bundle}\n------")
            print(f"node_with_scores_query_rewrite:\n{node_with_scores_query_rewrite}\n------")

        node_with_scores = merge_node(node_with_scores, node_with_scores_query_rewrite)

    ''' 长上下文阅读器 '''
    if cfg["LREORDER"]:
        LCreorder = LongContextReorder()
        node_with_scores = LCreorder.postprocess_nodes(node_with_scores)

    '''
    hyde: 基于假设，通过大语言模型生成的答案在Embedding空间中可能更为接近。HyDE通过生成假设性文档（答案）并利用Embedding相似性检索与之类似的真实文档来实现。
    '''
    if cfg["HYDE"]:
        print('Hyde')
        query_hyde = hydeGenerations(query_str=query_str, llm=llm)
        query_hyde_bundle = QueryBundle(query_str=query_hyde)

        node_with_scores_query_hyde = await retriever.aretrieve(query_hyde_bundle)

        if debug:
            print(f"query_hyde_bundle:\n{query_hyde_bundle}\n------")
            print(f"node_with_scores_query_hyde:\n{node_with_scores_query_hyde}\n------")

        node_with_scores = merge_node(node_with_scores, node_with_scores_query_hyde)

    if debug:
        print(f"retrieved:\n{node_with_scores}\n------")

    '''
    重排序
    '''
    if reranker:
        node_with_scores = reranker.postprocess_nodes(node_with_scores, query_bundle)

        if reranker_top is not None:
            node_with_scores = node_with_scores[:reranker_top]

        if debug:
            print(f"reranked:\n{node_with_scores}\n------")
    context_str = "\n\n".join(
        [f"{node.metadata['document_title']}: {node.text}" for node in node_with_scores]
    )
    fmt_qa_prompt = PromptTemplate(qa_template).format(
        context_str=context_str, query_str=query_str
    )
    print("-" * 50)
    print(fmt_qa_prompt)
    print("-" * 50)

    # print("llm----->:")
    # res = []
    # for node in node_with_scores:
    #     # print(node.metadata["file_name"])
    #     res.append(node.metadata["file_name"])
    #     print(node.text)
    #     print("-"*100)
    # print(res)

    ret = await llm.acomplete(fmt_qa_prompt)
    if progress:
        progress.update(1)
    return ret
