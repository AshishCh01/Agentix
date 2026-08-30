import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_logs import AgentLog
from app.models.session import ChatSession
from app.models.message import Message
from app.models.user import User
from sqlalchemy.dialects.postgresql import insert

# --- User Operations ---
async def sync_user(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    email: str,
    full_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> None:
    """
    Synchronizes a user from external auth into the local database using an upsert.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = insert(User).values(
        id=user_id,
        email=email,
        full_name=full_name,
        avatar_url=avatar_url
    ).on_conflict_do_update(
        index_elements=['id'],
        set_={
            'email': email,
            'full_name': full_name,
            'avatar_url': avatar_url
        }
    )
    await db.execute(stmt)
    await db.commit()



# --- Agent Execution Telemetry ---
async def log_agent_execution(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    node_name: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    execution_time_ms: float,
    model_used: Optional[str],
) -> AgentLog:
    """
    Logs agent execution metrics, latency, and inputs/outputs to the agent_logs table.
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    log_entry = AgentLog(
        session_id=session_id,
        agent_name=node_name,
        input_data=input_data,
        output_data=output_data,
        execution_time_ms=execution_time_ms,
    )
    db.add(log_entry)
    await db.commit()
    await db.refresh(log_entry)
    return log_entry


# --- Chat Session Operations ---
async def create_chat_session(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    title: str = "New Conversation",
) -> ChatSession:
    """
    Creates and persists a new chat session for a user.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_user_chat_sessions(
    db: AsyncSession,
    user_id: uuid.UUID | str,
) -> List[ChatSession]:
    """
    Fetches all chat sessions belonging to a specific user, ordered by most recent.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_chat_session(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
) -> Optional[ChatSession]:
    """
    Retrieves a single chat session by session_id, scoped to its owning
    user_id. Returns None if the session doesn't exist or belongs to a
    different user -- user_id is required (not optional) so this can never
    silently skip the ownership check and return another user's session.
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# Alias for backwards compatibility
get_chat_session_by_id = get_chat_session


async def update_chat_session_title(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
    title: str,
) -> Optional[ChatSession]:
    """
    Updates the title of an existing chat session.
    """
    session = await get_chat_session(db, session_id, user_id)
    if session:
        session.title = title
        await db.commit()
        await db.refresh(session)
    return session


async def delete_chat_session(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
) -> bool:
    """
    Deletes a chat session belonging to a user.
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = delete(ChatSession).where(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# --- Message Operations ---
async def get_session_messages(
    db: AsyncSession,
    session_id: uuid.UUID | str,
) -> List[Message]:
    """
    Retrieves conversation history messages for a chat session in chronological order.
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# Alias for backwards compatibility
get_messages_by_session = get_session_messages