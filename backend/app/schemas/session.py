import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class SessionCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class SessionUpdate(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)