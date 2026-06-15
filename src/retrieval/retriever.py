from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from ..utils.config import config
from ..utils.logger import logger


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


def retrieve(query: str) -> list:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    logger.info("Retrieved " + str(len(docs)) + " chunks")
    return docs
