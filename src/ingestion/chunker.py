from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..utils.config import config


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    out = []
    for doc in documents:
        parts = splitter.split_text(doc.page_content)
        for i, p in enumerate(parts):
            meta = dict(doc.metadata or {})
            meta["chunk_index"] = i
            out.append(Document(page_content=p, metadata=meta))
    return out
