import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from ai.services.retrieval_service import search_similar_chunks
from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.schemas.search import ChunkResult, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Vector Search"])


@router.post("", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def perform_vector_search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Performs cosine similarity vector search against stored document chunks
    for a given session_id.
    """
    user_id = str(current_user.get("id") or current_user.get("sub", ""))
    
    from app.database import crud
    try:
        session = await crud.get_chat_session(db, session_id=request.session_id, user_id=user_id)
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

    try:
        raw_chunks = await search_similar_chunks(
            db=db,
            query_text=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
        )

        # Convert dictionary list into ChunkResult objects for strict type checking
        chunk_models = [ChunkResult(**c) for c in raw_chunks]

        return SearchResponse(
            query=request.query,
            session_id=str(request.session_id),
            results_count=len(chunk_models),
            chunks=chunk_models,
        )
    except SQLAlchemyError as db_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error during vector similarity search.",
        )
    except Exception as e:
        logger.error("Vector similarity search failed: %s", e, exc_info=True)
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Embedding API rate limit exceeded.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector similarity search failed. Please try again later.",
        )