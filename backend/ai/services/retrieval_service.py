import uuid
from typing import Any, Dict, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.embedding_service import embedding_service
from app.models.document import Document
from app.models.document_chunks import DocumentChunk


async def search_similar_chunks(
    db: AsyncSession,
    query_text: str,
    session_id: uuid.UUID,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generates a vector embedding for the query string and performs a pgvector cosine
    distance search against document chunks filtered by chat session_id.
    """
    # 1. Generate 768-d embedding vector for the search query
    query_vector = embedding_service.generate_embedding(query_text)

    # 2. Define pgvector cosine distance calculation
    distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label(
        "distance"
    )

    # 3. Build query joining document chunks to documents to enforce session filtering
    stmt = (
        select(DocumentChunk, Document.filename, distance_col)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.session_id == session_id)
        .order_by(distance_col)
        .limit(top_k)
    )

    result = await db.execute(stmt)
    rows = result.all()

    # 4. Format and return retrieved context chunks
    retrieved_chunks = []
    for chunk, filename, distance in rows:
        # Convert cosine distance to a similarity score (1.0 - distance)
        similarity_score = round(1.0 - float(distance), 4)
        retrieved_chunks.append(
            {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "similarity_score": similarity_score,
            }
        )

    return retrieved_chunks