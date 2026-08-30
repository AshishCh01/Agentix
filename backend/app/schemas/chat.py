import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from app.config.settings import settings

# Base64 encoding inflates size ~4/3; allow up to MAX_UPLOAD_SIZE_MB of
# underlying image data (plus headroom for a "data:image/...;base64," prefix),
# matching the same configured limit used for document uploads.
_MAX_IMAGE_DATA_LENGTH = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 * 4 / 3) + 100


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    message: str = Field(default="", max_length=8000)
    image_data: Optional[str] = Field(default=None, max_length=_MAX_IMAGE_DATA_LENGTH)


class ChatSource(BaseModel):
    filename: str
    chunk_index: int
    score: float
    page_number: Optional[int] = None
    url: Optional[str] = None
    source_type: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    intent: Optional[str] = None
    sources: List[ChatSource] = []