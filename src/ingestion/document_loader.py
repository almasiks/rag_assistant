from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger


def load_pdf(file_path: Path) -> List[Document]:
    from pypdf import PdfReader

    docs = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning(f"  Пустая страница {page_num + 1} в {file_path.name}, пропускаем")
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page": page_num + 1,
                    "total_pages": len(reader.pages),
                },
            )
        )
    return docs

def load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    return [Document(page_content=text, metadata={
        "source": str(path),
        "file_name": path.name,
        "file_type": path.suffix.lstrip("."),
    })]


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def load_document(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return load_directory(path)
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"No loader for extension {suffix}")
    return loader(path)


def load_directory(path: Path) -> List[Document]:
    all_docs = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in LOADERS:
            all_docs.extend(load_document(p))
    return all_docs
