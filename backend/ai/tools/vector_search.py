import uuid
from typing import Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.retrieval_service import search_similar_chunks


async def vector_search_tool(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Tool function for agents to perform semantic vector search on uploaded documents.
    Retrieves top_k context chunks matching the user's query for a specific session.
    """
    retrieved_chunks = await search_similar_chunks(
        db=db,
        query_text=query,
        session_id=session_id,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return {
            "status": "empty",
            "message": "No relevant document chunks found for the given query.",
            "context_text": "",
            "chunks": [],
        }

    # Format context into a clean, combined string for LLM ingestion
    formatted_context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"[Source {idx}: {chunk['filename']} | Score: {chunk['similarity_score']}]\n"
            f"{chunk['content']}"
        )
        formatted_context_blocks.append(block)

    combined_context = "\n\n---\n\n".join(formatted_context_blocks)

    return {
        "status": "success",
        "results_count": len(retrieved_chunks),
        "context_text": combined_context,
        "chunks": retrieved_chunks,
    }