import base64
import logging
import uuid
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.llm_service import llm_service
from ai.services.retrieval_service import search_similar_chunks

logger = logging.getLogger(__name__)


async def vector_search_tool(
    db: AsyncSession,
    session_id: uuid.UUID,
    query: str,
    image_data: Optional[str] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Tool function for agents to perform semantic vector search on uploaded documents.
    Retrieves top_k context chunks matching the user's query for a specific session.
    Supports visual search query extraction when an image is provided.
    """
    search_query = query.strip()

    # If an image is provided, generate a text search query from the image content
    if image_data:
        try:
            base64_str = image_data.split(",")[-1] if "," in image_data else image_data
            img_bytes = base64.b64decode(base64_str)
            visual_description = await llm_service.describe_image(
                file_bytes=img_bytes,
                prompt=(
                    "Extract the key concepts, main subject matter, technical terms, "
                    "and readable text from this image in 1-2 concise search sentences "
                    "for document vector retrieval."
                ),
            )
            if visual_description:
                if search_query:
                    search_query = f"{search_query} {visual_description}"
                else:
                    search_query = visual_description
        except Exception as e:
            logger.warning(f"Failed to generate search query from image_data: {e}")

    # Fallback string if search_query is empty
    if not search_query:
        search_query = "document summary overview"

    retrieved_chunks = await search_similar_chunks(
        db=db,
        query_text=search_query,
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