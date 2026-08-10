import uuid
from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    message: str
    image_data: Optional[str] = None


class ChatSource(BaseModel):
    filename: str
    chunk_index: int
    similarity_score: float
    page_number: Optional[int] = None
    url: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    intent: Optional[str] = None
    sources: List[ChatSource] = []