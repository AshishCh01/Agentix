import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agents.graph import rag_graph
from ai.agents.state import AgentState
from app.auth.dependencies import get_current_user
from app.config.database import get_db
from app.models.message import Message
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource

router = APIRouter(prefix="/chat", tags=["Multi-Agent Chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = str(current_user.get("id") or current_user.get("sub", ""))
    session_id = str(request.session_id)

    # 1. Fetch recent chat history
    stmt = (
        select(Message)
        .where(Message.session_id == uuid.UUID(session_id))
        .order_by(Message.created_at.desc())
        .limit(6)
    )
    result = await db.execute(stmt)
    history_records = list(reversed(result.scalars().all()))
    chat_history = [{"role": msg.sender, "content": msg.content} for msg in history_records]

    # 2. Prepare initial state payload with explicit AgentState typing
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": request.message,
        "intent": None,
        "chat_history": chat_history,
        "retrieved_chunks": [],
        "formatted_context": "",
        "tool_outputs": [],
        "final_response": "",
        "error": None,
    }

    # 3. Save incoming user message
    db.add(
        Message(
            session_id=uuid.UUID(session_id),
            sender="user",
            content=request.message,
            citations=[],
        )
    )

    # 4. Invoke LangGraph Execution Engine
    try:
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": db}},
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph execution error: {str(e)}",
        )

    # 5. Extract sources & save assistant message
    sources = [
        ChatSource(
            filename=chunk.get("filename", ""),
            chunk_index=chunk.get("chunk_index", 0),
            similarity_score=chunk.get("similarity_score", 0.0),
        )
        for chunk in final_state.get("retrieved_chunks", [])
    ]

    db.add(
        Message(
            session_id=uuid.UUID(session_id),
            sender="assistant",
            content=final_state["final_response"],
            citations=[s.model_dump() for s in sources],
        )
    )
    await db.commit()

    return ChatResponse(
        session_id=session_id,
        user_message=request.message,
        assistant_message=final_state["final_response"],
        intent=final_state.get("intent"),
        sources=sources,
    )