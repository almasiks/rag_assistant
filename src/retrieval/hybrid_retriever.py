from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, SearchRequest, NamedVector, NamedSparseVector, FusionQuery, Fusion, Prefetch
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from ..utils.config import config
from ..utils.logger import logger

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL = "Qdrant/bm25"


def hybrid_retrieve(query: str, k: int = None) -> list[Document]:
    if k is None:
        k = config.TOP_K

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Dense vector
    dense_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    dense_vector = dense_model.embed_query(query)

    # Sparse BM25 vector
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vector = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    # Hybrid search with RRF fusion (built into Qdrant)
    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=k * 2),
            Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    docs = []
    for point in results.points:
        payload = point.payload or {}
        docs.append(Document(
            page_content=payload.get("page_content", ""),
            metadata={k: v for k, v in payload.items() if k != "page_content"},
        ))

    logger.info("Hybrid search returned " + str(len(docs)) + " docs for: " + query[:50])
    return docs
