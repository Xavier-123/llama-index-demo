import os

cfg = {}

# 多embedding
is_multi_embedding = os.environ.get("IS_MULTI_EMBEDDING", "F:\inspur\EMBEDDING_MODEL\m3e-base,F:\inspur\EMBEDDING_MODEL\m3e-small")
# is_multi_embedding = os.environ.get("IS_MULTI_EMBEDDING", "F:\inspur\EMBEDDING_MODEL\m3e-base")
embedding_list = is_multi_embedding.split(',')
cfg["EMBEDDING_LIST"] = embedding_list

# reranker
# cfg["RERANKER_MODEL"] = os.environ.get("RERANKER_MODEL", r"F:\inspur\EMBEDDING_MODEL\Xorbits\bge-reranker-base")
cfg["RERANKER_MODEL"] = os.environ.get("RERANKER_MODEL", False)

# quer重写
cfg["QUERY_REWRITE"] = os.environ.get("QUERY_REWRITE", False)

# 长内容优先排序
cfg["LREORDER"] = os.environ.get("LREORDER", False)

# 假设性文档嵌入
cfg["HYDE"] = os.environ.get("HYDE", False)

# 混合检索
cfg["MIXED_SEARCH"] = os.environ.get("MIXED_SEARCH", False)

# 重制作index
cfg["REINDEX"] = os.environ.get("REINDEX", False)   # True

cfg["VECTOR_SIZE"] = os.environ.get("VECTOR_SIZE", 768)

cfg["GLM_KEY"] = os.environ.get("GLM_KEY", '4674c23251e02b9c04b6d2c78e73a7a3.5azT9pxx0cHPX89Q')

cfg["DEBUG"] = os.environ.get("DEBUG", False)

# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-all")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-docx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-xlsx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-pdf")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-pptx")
cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-jcpt")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-swyy")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-yhfx")
# cfg["COLLECTION_NAME"] = os.environ.get("COLLECTION_NAME", "pdd-zcyy")
# cfg["DATA_DIR"] = os.environ.get("DATA_DIR", 'data/Product_Department_Data/')
cfg["DATA_DIR"] = os.environ.get("DATA_DIR", 'data/Product_Department_Data/基础平台/AI平台')
