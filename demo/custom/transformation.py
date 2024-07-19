from typing import Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core import PromptTemplate


class CustomFilePathExtractor(BaseExtractor):
    last_path_length: int = 4

    def __init__(self, last_path_length: int = 4, **kwargs):
        super().__init__(last_path_length=last_path_length, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFilePathExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            node.metadata["file_path"] = "/".join(
                node.metadata["file_path"].split("/")[-self.last_path_length:]
            )
            metadata_list.append(node.metadata)
        return metadata_list


class CustomTitleExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomTitleExtractor"

    # 将Document的第一行作为标题
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        try:
            document_title = nodes[0].text.split("\n")[0]
            last_file_path = nodes[0].metadata["file_path"]
        except:
            document_title = ""
            last_file_path = ""
        metadata_list = []
        for node in nodes:
            if node.metadata["file_path"] != last_file_path:
                document_title = node.text.split("\n")[0]
                last_file_path = node.metadata["file_path"]
            node.metadata["document_title"] = document_title
            metadata_list.append(node.metadata)

        return metadata_list


class CustomQueryEngine(BaseQueryEngine):
    def __init__(self,
                 llm,
                 retriever,
                 # filters,
                 qa_template,
                 reranker,
                 debug,
                 **kwargs):
        self.llm = llm
        self.retriever = retriever
        # self.filters=filters,
        self.qa_template = qa_template
        self.reranker = reranker
        self.debug = debug
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomQueryEngine"

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        node_with_scores = await self.retriever.aretrieve_cc(
            query_bundle,
            # qdrant_filters=self.filters
        )

        context_str = ''
        context_str += "\n".join(
            [f"背景知识{inode}:\n{node.text}" for inode, node in enumerate(node_with_scores)]
        )
        context_str = context_str.replace('~', '')
        context_str = context_str.replace('$', '')
        context_str = context_str.replace('>>>', '')
        context_str = context_str.replace('>>:', '')

        fmt_qa_prompt = PromptTemplate(self.qa_template).format(
            context_str=context_str, query_str=query_bundle.query_str
        )

        try:
            ret = await self.llm.acomplete(fmt_qa_prompt)
        except Exception as e:
            print(f'Request failed:{context_str}')
            ret = CompletionResponse(text='不确定')
        return Response(ret.text)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    def _get_prompt_modules(self):
        return {}


class CustomSentenceTransformerRerank():
    def __init__(self):
        pass
