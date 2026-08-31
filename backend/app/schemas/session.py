import uuid
from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    title: Optional[str] = "New Conversation"
    knowledge_base_id: Optional[uuid.UUID] = None


class SessionUpdate(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    knowledge_base_id: Optional[uuid.UUID] = None
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    sender: str
    content: str
    citations: Optional[List[Any]] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)