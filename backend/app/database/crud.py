import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_logs import AgentLog
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
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
    knowledge_base_id: uuid.UUID | str | None = None,
) -> ChatSession:
    """
    Creates and persists a new chat session for a user, optionally scoped to
    a knowledge base so RAG retrieval within it is filtered to that KB's
    documents.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)

    session = ChatSession(user_id=user_id, title=title, knowledge_base_id=knowledge_base_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_user_chat_sessions(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    limit: int = 50,
    offset: int = 0,
) -> List[ChatSession]:
    """
    Fetches a page of chat sessions belonging to a specific user, ordered by
    most recent.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
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
    limit: int = 50,
    offset: int = 0,
) -> List[Message]:
    """
    Retrieves a page of conversation history messages for a chat session in
    chronological order.
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# Alias for backwards compatibility
get_messages_by_session = get_session_messages


# --- Knowledge Base Operations ---
async def create_knowledge_base(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    name: str,
    description: Optional[str] = None,
) -> KnowledgeBase:
    """
    Creates and persists a new knowledge base for a user.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    kb = KnowledgeBase(user_id=user_id, name=name, description=description)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


async def get_user_knowledge_bases(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    limit: int = 100,
    offset: int = 0,
) -> List[tuple[KnowledgeBase, int]]:
    """
    Fetches a page of knowledge bases belonging to a user, most recently
    updated first, each paired with its current document count.
    """
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = (
        select(KnowledgeBase, func.count(Document.id))
        .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
        .where(KnowledgeBase.user_id == user_id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return [(kb, count) for kb, count in result.all()]


async def get_knowledge_base(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
) -> Optional[KnowledgeBase]:
    """
    Retrieves a single knowledge base by id, scoped to its owning user_id.
    Returns None if it doesn't exist or belongs to a different user --
    user_id is required (not optional) so this can never silently skip the
    ownership check and return another user's knowledge base.
    """
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = select(KnowledgeBase).where(
        KnowledgeBase.id == knowledge_base_id, KnowledgeBase.user_id == user_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_knowledge_base_document_count(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
) -> int:
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)

    stmt = select(func.count(Document.id)).where(
        Document.knowledge_base_id == knowledge_base_id
    )
    result = await db.execute(stmt)
    return result.scalar_one()


async def update_knowledge_base(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[KnowledgeBase]:
    """
    Updates the name and/or description of an existing knowledge base.
    Fields left as None are not modified.
    """
    kb = await get_knowledge_base(db, knowledge_base_id, user_id)
    if kb:
        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        await db.commit()
        await db.refresh(kb)
    return kb


async def delete_knowledge_base(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
    user_id: uuid.UUID | str,
) -> bool:
    """
    Deletes a knowledge base belonging to a user. Documents attached to it
    are cascade-deleted at the database level.
    """
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    stmt = delete(KnowledgeBase).where(
        KnowledgeBase.id == knowledge_base_id, KnowledgeBase.user_id == user_id
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


async def get_knowledge_base_documents(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
    limit: int = 100,
    offset: int = 0,
) -> List[Document]:
    """
    Fetches a page of documents belonging to a knowledge base, most recently
    uploaded first. Ownership of the knowledge base must be verified by the
    caller (see get_knowledge_base) before calling this.
    """
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)

    stmt = (
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_knowledge_base_document(
    db: AsyncSession,
    knowledge_base_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
) -> Optional[Document]:
    """
    Retrieves a single document scoped to a specific knowledge base.
    Ownership of the knowledge base must be verified by the caller before
    calling this.
    """
    if isinstance(knowledge_base_id, str):
        knowledge_base_id = uuid.UUID(knowledge_base_id)
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)

    stmt = select(Document).where(
        Document.id == document_id, Document.knowledge_base_id == knowledge_base_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def delete_document(db: AsyncSession, document_id: uuid.UUID | str) -> bool:
    """
    Deletes a document (and its chunks, via cascade). Ownership/scope must
    be verified by the caller before calling this.
    """
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)

    stmt = delete(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0