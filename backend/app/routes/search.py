from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.retrieval_service import search_similar_chunks
from app.auth.dependencies import get_current_user
from app.config.database import get_db
from app.schemas.search import ChunkResult, SearchRequest, SearchResponse

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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector similarity search failed: {str(e)}",
        )