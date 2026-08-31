import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: uuid.UUID
    status: str
    filename: str


class DocumentResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    session_id: Optional[uuid.UUID] = None
    knowledge_base_id: Optional[uuid.UUID] = None
    filename: str
    file_type: str
    file_path: str
    file_size: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]