import os

cfg = {}

# 多embedding
is_multi_embedding = os.environ.get("IS_MULTI_EMBEDDING", "F:\inspur\EMBEDDING_MODEL\m3e-base,F:\inspur\EMBEDDING_MODEL\m3e-small")
embedding_list = is_multi_embedding.split(',')
cfg["EMBEDDING_LIST"] = embedding_list
if len(embedding_list) > 1:
    embedding_model1 = embedding_list[0]
    embedding_model2 = embedding_list[-1]
    cfg["EMBEDDING_MODEL1"] = embedding_model1
    cfg["EMBEDDING_MODEL2"] = embedding_model2
else:
    embedding_model1 = is_multi_embedding
    cfg["EMBEDDING_MODEL1"] = embedding_model1

# reranker
cfg["RERANKER_MODEL"] = os.environ.get("RERANKER_MODEL", r"F:\inspur\EMBEDDING_MODEL\Xorbits\bge-reranker-base")

# quer重写
cfg["QUERY_REWRITE"] = os.environ.get("QUERY_REWRITE", False)

# 长内容优先排序
cfg["LREORDER"] = os.environ.get("LREORDER", False)

# 假设性文档嵌入
cfg["HYDE"] = os.environ.get("HYDE", False)

# 混合检索
cfg["MIXED_SEARCH"] = os.environ.get("MIXED_SEARCH", False)

# 重制作index
cfg["REINDEX"] = os.environ.get("REINDEX", True)   # True

cfg["VECTOR_SIZE"] = os.environ.get("VECTOR_SIZE", 768)

cfg["GLM_KEY"] = os.environ.get("GLM_KEY", '4674c23251e02b9c04b6d2c78e73a7a3.5azT9pxx0cHPX89Q')

# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "md")
# cfg["DATA_DIR"] = os.environ.get("DATA_DIR", 'data/md')

cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-docx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-xlsx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-pdf")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-pptx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-zip")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-rar")
cfg["DATA_DIR"] = os.environ.get("DATA_DIR", 'data/Product_Department_Data')
