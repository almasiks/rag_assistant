import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse, SourceInfo
from .database import init_db, log_query
from ..retrieval.hybrid_retriever import hybrid_retrieve
from ..retrieval.chain import ask
from ..utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("RAG API started (hybrid mode)")
    yield
    logger.info("RAG API stopped")


app = FastAPI(title="RAG API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "hybrid"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    docs = hybrid_retrieve(request.question)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    answer = ask(request.question, docs)
    latency_ms = int((time.time() - start) * 1000)

    sources = [
        SourceInfo(
            file_name=doc.metadata.get("file_name", "unknown"),
            page=doc.metadata.get("page"),
            chunk_index=doc.metadata.get("chunk_index"),
        )
        for doc in docs
    ]

    await log_query(request.question, answer, [s.file_name for s in sources], latency_ms)
    logger.info("Query done in " + str(latency_ms) + "ms")
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)
