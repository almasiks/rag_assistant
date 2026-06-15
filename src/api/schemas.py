from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    file_name: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    latency_ms: int
