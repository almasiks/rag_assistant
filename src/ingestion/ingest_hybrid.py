import argparse
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
import uuid
from ..ingestion.document_loader import load_directory, load_document
from ..ingestion.chunker import split_documents
from ..utils.config import config
from ..utils.logger import logger


SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def get_or_create_hybrid_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME in collections:
        logger.info("Collection exists, deleting and recreating for hybrid...")
        client.delete_collection(config.QDRANT_COLLECTION_NAME)

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    logger.info("Hybrid collection created")


def ingest_hybrid(path: Path):
    if path.is_dir():
        docs = load_directory(path)
    else:
        docs = load_document(path)

    chunks = split_documents(docs)
    logger.info("Total chunks: " + str(len(chunks)))

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    get_or_create_hybrid_collection(client)

    # Dense embeddings
    logger.info("Computing dense embeddings...")
    dense_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    texts = [c.page_content for c in chunks]
    dense_vectors = dense_model.embed_documents(texts)

    # Sparse BM25 embeddings
    logger.info("Computing sparse BM25 embeddings...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_vectors = list(sparse_model.embed(texts))

    # Upload points
    logger.info("Uploading points to Qdrant...")
    points = []
    for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload={
                "page_content": chunk.page_content,
                **chunk.metadata,
            },
        ))

    client.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=points)
    final = client.get_collection(config.QDRANT_COLLECTION_NAME).points_count
    logger.info("Done! Points in collection: " + str(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    ingest_hybrid(args.path)
