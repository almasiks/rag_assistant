import argparse
from pathlib import Path
from ..utils.logger import logger
from .document_loader import load_document
from .chunker import split_documents
from .embedder import store_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Only show first chunks, do not write to Qdrant")
    args = parser.parse_args()

    p = Path(args.path)
    logger.info("Loading documents from %s", p)
    docs = load_document(p)
    logger.info("Loaded %d documents", len(docs))

    chunks = split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    if args.dry_run:
        logger.info("Dry run enabled — printing first 3 chunks:")
        for i, c in enumerate(chunks[:3]):
            print(f"--- CHUNK {i} ---")
            print(c.page_content[:1000])
            print(c.metadata)
        return

    store_chunks(chunks)


if __name__ == "__main__":
    main()
