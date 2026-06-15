from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from ..utils.config import config
from ..utils.logger import logger


def get_embedding_model():
    logger.info("Loading model: " + config.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_qdrant_client():
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def ensure_collection_exists(client):
    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )


def store_chunks(chunks):
    if not chunks:
        raise ValueError("No chunks")
    client = get_qdrant_client()
    ensure_collection_exists(client)
    embeddings = get_embedding_model()
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://" + config.QDRANT_HOST + ":" + str(config.QDRANT_PORT),
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    return vectorstore
