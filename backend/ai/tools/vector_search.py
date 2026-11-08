import logging
import uuid
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.retrieval_service import search_similar_chunks

logger = logging.getLogger(__name__)


async def vector_search_tool(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    query: str,
    image_data: Optional[str] = None,
    top_k: int = 4,
    knowledge_base_id: Optional[uuid.UUID | str] = None,
    user_id: Optional[uuid.UUID | str] = None,
) -> Dict[str, Any]:
    """
    Performs Hybrid Search (Dense pgvector + Sparse TSVector full-text) with
    Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking via
    ai.services.retrieval_service.search_similar_chunks, then formats the
    result as LLM-ready context. When knowledge_base_id is set, retrieval includes
    both documents from that knowledge base and documents uploaded directly to the
    chat session.

    If query is empty and image_data is provided, resolves the image to a textual
    search query before executing retrieval.
    """
    if not user_id:
        logger.warning("vector_search_tool called without user_id; failing closed.")
        return {"chunks": [], "context_text": ""}

    if isinstance(user_id, str):
        try:
            uuid.UUID(user_id)
        except (ValueError, TypeError):
            logger.warning("vector_search_tool called with invalid user_id '%s'; failing closed.", user_id)
            return {"chunks": [], "context_text": ""}
    elif not isinstance(user_id, uuid.UUID):
        logger.warning("vector_search_tool called with non-UUID user_id type '%s'; failing closed.", type(user_id))
        return {"chunks": [], "context_text": ""}

    # Fallback: if query is empty and image_data is provided, resolve image into text query
    if not query and image_data:
        from ai.services.vision_service import resolve_image_query
        resolved_q, err = await resolve_image_query(image_data=image_data, user_query="")
        if resolved_q:
            query = resolved_q
        else:
            logger.warning("vector_search_tool could not resolve image query: %s", err)
            return {"chunks": [], "context_text": ""}

    try:
        chunks = await search_similar_chunks(
            db=db,
            query_text=query,
            session_id=session_id,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"❌ Hybrid Search execution error: {str(e)}")
        return {"chunks": [], "context_text": "", "retrieval_error": str(e)}

    if not chunks:
        return {"chunks": [], "context_text": ""}

    formatted_chunks = [
        {
            "id": c.get("chunk_id"),
            "content": c.get("content"),
            "page_number": c.get("page_number") or 1,
            "chunk_index": c.get("chunk_index"),
            "filename": c.get("filename"),
            "score": c.get("score"),
            "source_type": c.get("source_type"),
            "rerank_score": c.get("rerank_score"),
            "rerank_logit": c.get("rerank_logit"),
        }
        for c in chunks
    ]

    context_blocks = [
        f"[Source: {c['filename']} | Page {c['page_number']}]\n{c['content']}"
        for c in formatted_chunks
    ]
    formatted_context = "\n\n---\n\n".join(context_blocks)

    return {
        "chunks": formatted_chunks,
        "context_text": formatted_context,
    }
