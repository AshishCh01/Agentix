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
) -> Dict[str, Any]:
    """
    Performs Hybrid Search (Dense pgvector + Sparse TSVector full-text) with
    Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking via
    ai.services.retrieval_service.search_similar_chunks, then formats the
    result as LLM-ready context. When knowledge_base_id is set, retrieval is
    scoped to that knowledge base instead of the chat session.
    """
    try:
        chunks = await search_similar_chunks(
            db=db,
            query_text=query,
            session_id=session_id,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
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
