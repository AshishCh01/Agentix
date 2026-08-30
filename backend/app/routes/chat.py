import json
import logging
import time
import uuid
from typing import AsyncGenerator, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agents.graph import rag_graph
from ai.agents.state import AgentState
from ai.services.llm_service import extract_text_from_content, llm_service
from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.database.crud import log_agent_execution
from app.models.message import Message
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Multi-Agent Chat"])


async def _setup_chat_state(
    request: ChatRequest,
    current_user: dict,
    db: AsyncSession,
) -> Tuple[str, str, AgentState]:
    """
    Helper function to consolidate history fetching, user message persistence,
    and initial state setup across both /chat and /chat/stream endpoints.
    """
    user_id = str(current_user.get("id") or current_user.get("sub", ""))
    session_id = str(request.session_id)
    user_query_text = request.message or ""

    from app.database import crud
    try:
        session = await crud.get_chat_session(db, session_id=session_id, user_id=user_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found or access denied.",
            )
    except SQLAlchemyError as db_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while verifying session.",
        )

    # Fetch recent history
    try:
        stmt = (
            select(Message)
            .where(Message.session_id == uuid.UUID(session_id))
            .order_by(Message.created_at.desc())
            .limit(6)
        )
        result = await db.execute(stmt)
        history_records = list(reversed(result.scalars().all()))
        chat_history = [{"role": msg.sender, "content": msg.content} for msg in history_records]
    except SQLAlchemyError as db_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while fetching chat history.",
        )

    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "user_query": user_query_text,
        "image_data": request.image_data,
        "intent": None,
        "chat_history": chat_history,
        "retrieved_chunks": [],
        "formatted_context": "",
        "context_source": None,
        "tool_outputs": [],
        "final_response": "",
        "error": None,
        "retry_count": 0,
    }

    # Persist user message
    db.add(
        Message(
            session_id=uuid.UUID(session_id),
            sender="user",
            content=user_query_text if user_query_text else "[Image Attached]",
            citations=[],
        )
    )
    await db.commit()

    return session_id, user_query_text, initial_state


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    start_time = time.perf_counter()
    session_id, user_query_text, initial_state = await _setup_chat_state(
        request, current_user, db
    )

    try:
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": db}},
        )
    except SQLAlchemyError as db_err:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error during LangGraph execution.",
        )
    except Exception as e:
        await db.rollback()
        logger.error("LangGraph execution error in /chat: %s", e, exc_info=True)
        error_msg = str(e).lower()
        if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM rate limit exceeded. Please try again later.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to process your request. Please try again later.",
        )

    sources = [
        ChatSource(
            filename=chunk.get("filename", ""),
            chunk_index=chunk.get("chunk_index", 0),
            score=chunk.get("score", 0.0),
            page_number=chunk.get("page_number"),  # Added for PDF page mapping
            url=chunk.get("url"),                  # Added for Web search mapping
            source_type=chunk.get("source_type")
        )
        for chunk in final_state.get("retrieved_chunks", [])
    ]

    clean_answer = extract_text_from_content(final_state.get("final_response", ""))

    db.add(
        Message(
            session_id=uuid.UUID(session_id),
            sender="assistant",
            content=clean_answer,
            citations=[s.model_dump() for s in sources],
        )
    )
    await db.commit()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Log Execution Telemetry to agent_logs table
    await log_agent_execution(
        db=db,
        session_id=uuid.UUID(session_id),
        node_name=final_state.get("intent") or "supervisor",
        input_data={"user_query": user_query_text},
        output_data={
            "final_response": clean_answer,
            "sources": [s.model_dump() for s in sources],
        },
        execution_time_ms=elapsed_ms,
        model_used=llm_service.default_model,
    )

    return ChatResponse(
        session_id=session_id,
        user_message=user_query_text,
        assistant_message=clean_answer,
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
    Server-Sent Events (SSE) endpoint that streams node updates, real-time LLM token deltas,
    and the final completion payload with reflection metrics.
    """
    session_id, user_query_text, initial_state = await _setup_chat_state(request, current_user, db)

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        final_state: dict = {}
        try:
            async for event in rag_graph.astream_events(
                initial_state,
                version="v2",
                config={"configurable": {"db": db}},
            ):
                kind = event.get("event")
                node_name = event.get("name", "")

                # 1. Active Node transitions
                if kind == "on_chain_start" and node_name in [
                    "supervisor",
                    "vector_search",
                    "web_search",
                    "answer",
                    "greeting",
                    "reflection",
                ]:
                    payload = json.dumps({"type": "node_start", "node": node_name})
                    yield f"data: {payload}\n\n"

                # 2. Real-time token streaming from LLM calls
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    raw_content = getattr(chunk, "content", "") if chunk else ""
                    token_content = extract_text_from_content(raw_content)

                    if token_content:
                        token_payload = json.dumps({"type": "token", "content": token_content})
                        yield f"data: {token_payload}\n\n"

                # 3. Capture State Output on Graph Completion
                elif kind == "on_chain_end" and node_name in ["LangGraph", "rag_graph"]:
                    final_state = event.get("data", {}).get("output", {})

            clean_final_response = extract_text_from_content(final_state.get("final_response", ""))
            intent = final_state.get("intent", "RAG_QUERY")
            retrieved_chunks = final_state.get("retrieved_chunks", [])

            sources = [
                {
                    "filename": chunk.get("filename", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "score": chunk.get("score", 0.0),
                    "page_number": chunk.get("page_number"),  # Added for PDF page mapping
                    "url": chunk.get("url"),                  # Added for Web search mapping
                    "source_type": chunk.get("source_type")
                }
                for chunk in retrieved_chunks
            ]

            # Extract reflection metrics if present
            tool_outputs = final_state.get("tool_outputs", [])
            reflection_data = next(
                (t for t in tool_outputs if isinstance(t, dict) and t.get("node") == "reflection"),
                None,
            )

            # Persist Assistant Message to DB
            db.add(
                Message(
                    session_id=uuid.UUID(session_id),
                    sender="assistant",
                    content=clean_final_response,
                    citations=sources,
                )
            )
            await db.commit()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log Agent Execution to agent_logs
            await log_agent_execution(
                db=db,
                session_id=uuid.UUID(session_id),
                node_name=intent or "supervisor",
                input_data={"user_query": user_query_text},
                output_data={
                    "final_response": clean_final_response,
                    "sources": sources,
                },
                execution_time_ms=elapsed_ms,
                model_used=llm_service.default_model,
            )

            # 4. Stream Final Completion Payload with Reflection
            completion_payload = json.dumps({
                "type": "completion",
                "session_id": session_id,
                "intent": intent,
                "assistant_message": clean_final_response,
                "sources": sources,
                "reflection": reflection_data,
            })
            yield f"data: {completion_payload}\n\n"

        except SQLAlchemyError as db_err:
            await db.rollback()
            error_payload = json.dumps({"type": "error", "error_type": "database_error", "detail": "A database error occurred."})
            yield f"data: {error_payload}\n\n"
        except Exception as e:
            await db.rollback()
            logger.error("LangGraph execution error in /chat/stream: %s", e, exc_info=True)
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                error_type = "rate_limit_error"
                detail = "LLM rate limit exceeded. Please try again later."
            else:
                error_type = "execution_error"
                detail = "Failed to process your request. Please try again later."
            error_payload = json.dumps({"type": "error", "error_type": error_type, "detail": detail})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")