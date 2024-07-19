from typing import Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode, QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE


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
    def __init__(self, llm, retriever,
                 # filters,
                 qa_template, reranker, debug, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm,
        self.retriever = retriever,
        # self.filters=filters,
        self.qa_template = qa_template,
        self.reranker = reranker,
        self.debug = debug,
        self.callback_manager = None,

    @classmethod
    def class_name(cls) -> str:
        return "CustomQueryEngine"

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        node_with_scores = await self.retriever.aretrieve_cc(query_bundle, qdrant_filters=self.filters)
        print("_aquery:", node_with_scores)
        pass

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    def _get_prompt_modules(self):
        return {}


class CustomSentenceTransformerRerank():
    def __init__(self):
        pass
