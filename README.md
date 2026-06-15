# rag-system — Phase 1: Ingestion Pipeline

Run a dry-run to see chunks:

```bash
python -m src.ingestion.ingest docs/ --dry-run
```

To push to Qdrant (ensure Qdrant is running, e.g. `docker-compose up -d`):

```bash
python -m src.ingestion.ingest docs/
```
