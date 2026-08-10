import uuid
from typing import List
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    session_id: uuid.UUID = Field(
        ..., description="UUID of the chat session to search within"
    )
    query: str = Field(
        ..., min_length=1, description="Natural language search query string"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of top similar context chunks to retrieve",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    chunk_index: int
    content: str
    score: float
    source_type: str


class SearchResponse(BaseModel):
    query: str
    session_id: str
    results_count: int
    chunks: List[ChunkResult]