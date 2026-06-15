from dataclasses import dataclass
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()


@dataclass
class Config:
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag")
    DATA_RAW: Path = Path(os.getenv("DATA_RAW", "data/raw"))
    DATA_PROCESSED: Path = Path(os.getenv("DATA_PROCESSED", "data/processed"))


config = Config()
