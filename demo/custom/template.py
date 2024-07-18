from llama_index.core import PromptTemplate


QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，回答以下问题，可以分点作答，如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息：
    {query_str}

    回答：\
    """


SUMMARY_EXTRACT_TEMPLATE = """\
    这是这一小节的内容：
    {context_str}
    请用中文总结本节的关键主题和实体。

    总结：\
    """


RW_TEMPLATE = """\
    你是一个运维领域的专家，请将"{query_str}"这个问题拆分为两个子问题，问题中的专有名词需要保持不变，子问题之间用"\n"分割，问题：
    """

OW_TEMPLATE = """\
    你是一个运维领域的专家，请将"{query_str}"这个句话换成另一种专业提问方式，问题中的专有名词需要保持不变，提问方式：
    """
