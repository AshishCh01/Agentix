import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.session import ChatSession


async def create_chat_session(
    db: AsyncSession, user_id: uuid.UUID, title: str = "New Conversation"
) -> ChatSession:
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_user_chat_sessions(
    db: AsyncSession, user_id: uuid.UUID
) -> List[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_chat_session_by_id(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def update_chat_session_title(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID, title: str
) -> Optional[ChatSession]:
    session = await get_chat_session_by_id(db, session_id, user_id)
    if session:
        session.title = title
        await db.commit()
        await db.refresh(session)
    return session


async def delete_chat_session(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    session = await get_chat_session_by_id(db, session_id, user_id)
    if session:
        await db.delete(session)
        await db.commit()
        return True
    return False