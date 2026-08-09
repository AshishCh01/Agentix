import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
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
    user_query_text = request.message or ""

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

    # 2. Prepare initial state payload with image_data
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": user_query_text,
        "image_data": request.image_data,
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
            content=user_query_text if user_query_text else "[Image Attached]",
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
            content=final_state.get("final_response", ""),
            citations=[s.model_dump() for s in sources],
        )
    )
    await db.commit()

    return ChatResponse(
        session_id=session_id,
        user_message=user_query_text,
        assistant_message=final_state.get("final_response", ""),
        intent=final_state.get("intent"),
        sources=sources,
    )


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Server-Sent Events (SSE) streaming endpoint that streams node updates and tokens
    in real-time as LangGraph executes.
    """
    user_id = str(current_user.get("id") or current_user.get("sub", ""))
    session_id = str(request.session_id)
    user_query_text = request.message or ""

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

    # 2. Build initial state with image_data
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": user_query_text,
        "image_data": request.image_data,
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
            content=user_query_text if user_query_text else "[Image Attached]",
            citations=[],
        )
    )
    await db.commit()

    async def event_generator() -> AsyncGenerator[str, None]:
        final_state: dict = {}
        try:
            # Stream node transition events from LangGraph
            async for event in rag_graph.astream_events(
                initial_state,
                version="v2",
                config={"configurable": {"db": db}},
            ):
                kind = event.get("event")
                node_name = event.get("name", "")

                if kind == "on_chain_start" and node_name in ["supervisor", "vector_search", "web_search", "answer", "reflection"]:
                    payload = json.dumps({"type": "node_start", "node": node_name})
                    yield f"data: {payload}\n\n"

                elif kind == "on_chain_end" and node_name == "LangGraph":
                    final_state = event.get("data", {}).get("output", {})

            # Stream final aggregated payload
            final_response_text = final_state.get("final_response", "")
            intent = final_state.get("intent", "RAG_QUERY")
            retrieved_chunks = final_state.get("retrieved_chunks", [])

            sources = [
                {
                    "filename": chunk.get("filename", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "similarity_score": chunk.get("similarity_score", 0.0),
                }
                for chunk in retrieved_chunks
            ]

            # Save assistant message to DB
            db.add(
                Message(
                    session_id=uuid.UUID(session_id),
                    sender="assistant",
                    content=final_response_text,
                    citations=sources,
                )
            )
            await db.commit()

            completion_payload = json.dumps({
                "type": "completion",
                "session_id": session_id,
                "intent": intent,
                "assistant_message": final_response_text,
                "sources": sources,
            })
            yield f"data: {completion_payload}\n\n"

        except Exception as e:
            error_payload = json.dumps({"type": "error", "detail": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")