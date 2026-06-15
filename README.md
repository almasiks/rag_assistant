# RAG Assistant

A local Retrieval-Augmented Generation (RAG) system for document analysis. Upload your documents, ask questions, get answers with sources — all running locally, no external APIs required.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.13+-green)
![Qdrant](https://img.shields.io/badge/Qdrant-latest-red)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-orange)
![Docker](https://img.shields.io/badge/Docker-required-blue)

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development Phases](#development-phases)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI :8501                   │
│         Chat  │  Upload Documents  │  Statistics        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI :8000                         │
│              POST /query   GET /health                  │
└──────┬───────────────────────────────────┬──────────────┘
       │                                   │
┌──────▼──────┐                   ┌────────▼────────┐
│   Qdrant    │                   │     SQLite      │
│  :6333      │                   │   logs.db       │
│             │                   │                 │
│ Dense 384d  │                   │  query_logs     │
│ Sparse BM25 │                   │  (question,     │
│ RRF Fusion  │                   │  answer,        │
└─────────────┘                   │  latency_ms)    │
       │                          └─────────────────┘
┌──────▼──────┐
│   Ollama    │
│  :11434     │
│  llama3.2   │
└─────────────┘
```

**Query flow:**
```
User question
    → Qdrant hybrid search (dense + BM25 sparse + RRF fusion)
    → Top-5 relevant chunks
    → Ollama llama3.2 (prompt + context)
    → Answer with sources + logged to SQLite
```

**Ingestion flow:**
```
Documents (PDF / TXT / MD)
    → PyPDF / text loader
    → RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
    → HuggingFace all-MiniLM-L6-v2 (dense embeddings, 384d)
    → Qdrant BM25 (sparse embeddings)
    → Qdrant collection
```

---

## Features

- **Hybrid Search** — combines dense semantic search + BM25 sparse search with Reciprocal Rank Fusion (RRF) for better retrieval quality
- **Local LLM** — runs entirely on your machine via Ollama, no API keys needed
- **Multi-format ingestion** — supports PDF, TXT, and Markdown files
- **Streamlit UI** — chat interface, document uploader, and statistics dashboard
- **REST API** — FastAPI backend with Swagger docs at `/docs`
- **Query logging** — every request logged to SQLite with question, answer, sources, latency

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Store | [Qdrant](https://qdrant.tech/) |
| Embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Sparse Search | Qdrant BM25 via [FastEmbed](https://github.com/qdrant/fastembed) |
| LLM | [Ollama](https://ollama.com/) + llama3.2 |
| RAG Framework | [LangChain](https://langchain.com/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) |
| UI | [Streamlit](https://streamlit.io/) |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |
| Logging | SQLite + [loguru](https://github.com/Delgan/loguru) |
| Infrastructure | Docker + Docker Compose |

---

## Project Structure

```
rag_assistant/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # PDF, TXT, MD loaders
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter
│   │   ├── embedder.py           # Dense embeddings + Qdrant (Phase 1)
│   │   ├── ingest.py             # CLI entrypoint (dense only)
│   │   └── ingest_hybrid.py      # CLI entrypoint (hybrid: dense + BM25)
│   ├── retrieval/
│   │   ├── retriever.py          # Dense-only retriever
│   │   ├── hybrid_retriever.py   # Hybrid retriever with RRF fusion
│   │   └── chain.py              # LLM chain (Ollama + prompt)
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── database.py           # SQLite query logging
│   └── utils/
│       ├── config.py             # Settings from .env
│       └── logger.py             # Loguru setup
├── docs/                         # Put your documents here
├── data/
│   ├── raw/                      # Raw documents (optional)
│   └── logs.db                   # SQLite query log (auto-created)
├── app.py                        # Streamlit UI
├── docker-compose.yml            # Qdrant + PostgreSQL
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/download)

### 1. Clone and install

```bash
git clone https://github.com/almasiks/rag_assistant.git
cd rag_assistant

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# defaults are fine for local development
```

### 3. Start infrastructure

```bash
docker-compose up -d
# Qdrant dashboard: http://localhost:6333/dashboard
```

### 4. Pull LLM model

```bash
ollama pull llama3.2
```

### 5. Add documents and index

```bash
# Put your PDF/TXT/MD files in docs/
python -m src.ingestion.ingest_hybrid docs/
```

### 6. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Start the UI

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Usage

### Streamlit UI

Open `http://localhost:8501` in your browser:

- **Chat tab** — ask questions about your documents, see answers with sources and latency
- **Upload tab** — drag and drop PDF/TXT/MD files, they get indexed automatically
- **Statistics tab** — view total queries, average latency, query history chart

### CLI ingestion

```bash
# Index a single file
python -m src.ingestion.ingest_hybrid docs/my_doc.pdf

# Index a directory
python -m src.ingestion.ingest_hybrid docs/

# Dry run — preview chunks without writing to Qdrant
python -m src.ingestion.ingest docs/ --dry-run
```

---

## API Reference

### `GET /health`
```json
{"status": "ok", "mode": "hybrid"}
```

### `POST /query`

**Request:**
```json
{
  "question": "What is the difference between L1 and L2 regularization?"
}
```

**Response:**
```json
{
  "answer": "L1 regularization (Lasso) produces sparse models by pushing weights to zero, whereas L2 regularization (Ridge) shrinks weights evenly...",
  "sources": [
    {"file_name": "ml_guide.md", "page": null, "chunk_index": 5},
    {"file_name": "ml_guide.md", "page": null, "chunk_index": 2}
  ],
  "latency_ms": 10095
}
```

Interactive docs: `http://localhost:8000/docs`

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Ingestion pipeline: document loading, chunking, dense embeddings → Qdrant | ✅ Done |
| **Phase 2** | FastAPI + dense retrieval + Ollama LLM + SQLite logging | ✅ Done |
| **Phase 3** | Hybrid search: BM25 sparse vectors + RRF fusion in Qdrant | ✅ Done |
| **Phase 4** | Streamlit UI: chat, document upload, statistics dashboard | ✅ Done |

---

## Configuration

All settings in `.env`:

```env
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Retrieval
TOP_K=5
```

---

## License

MIT
