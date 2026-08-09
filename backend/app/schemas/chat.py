import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: uuid.UUID = Field(
        ..., description="UUID of the chat session"
    )
    message: Optional[str] = Field(
        "", description="User prompt or question message"
    )
    image_data: Optional[str] = Field(
        None, description="Base64 encoded image string (e.g., data:image/png;base64,...)"
    )


class ChatSource(BaseModel):
    filename: str
    chunk_index: int
    similarity_score: float


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    intent: Optional[str] = None
    sources: List[ChatSource] = []