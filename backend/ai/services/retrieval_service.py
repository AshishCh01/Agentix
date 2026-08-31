import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.embedding_service import embedding_service, EmbeddingTask
from app.models.document import Document
from app.models.document_chunks import DocumentChunk


async def search_similar_chunks(
    db: AsyncSession,
    query_text: str,
    session_id: uuid.UUID,
    top_k: int = 4,
    knowledge_base_id: Optional[uuid.UUID] = None,
) -> List[Dict[str, Any]]:
    """
    Performs hybrid search (Dense pgvector + Sparse FTS) with Reciprocal Rank Fusion (RRF),
    then reranks the top results using a cross-encoder.

    Scoping: when `knowledge_base_id` is provided, retrieval is restricted to
    documents belonging to that knowledge base (d.knowledge_base_id) rather
    than the chat session -- this is how a chat started against a selected
    Knowledge Base only ever retrieves that KB's documents. When it is
    omitted (the existing/default behavior for sessions with no KB
    attached), retrieval is scoped to `session_id` exactly as before.
    """
    from sqlalchemy import text

    # 1. Generate embedding vector for the search query
    query_vector = await embedding_service.generate_embedding(query_text, mode=EmbeddingTask.QUERY)
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    if knowledge_base_id is not None:
        scope_filter = "d.knowledge_base_id = :scope_id"
        scope_id = str(knowledge_base_id)
    else:
        scope_filter = "d.session_id = :scope_id"
        scope_id = str(session_id)

    # 2. Hybrid SQL Query with RRF
    hybrid_sql = f"""
    WITH sparse AS (
        SELECT
            dc.id,
            RANK() OVER (ORDER BY ts_rank_cd(dc.fts_tokens, websearch_to_tsquery('english', :query)) DESC) as rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE {scope_filter}
          AND dc.fts_tokens @@ websearch_to_tsquery('english', :query)
        LIMIT 20
    ),
    dense AS (
        SELECT
            dc.id,
            RANK() OVER (ORDER BY dc.embedding <=> CAST(:query_vector AS vector)) as rank
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE {scope_filter}
        LIMIT 20
    )
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.chunk_index,
        dc.page_number,
        dc.content,
        d.filename,
        COALESCE(1.0 / (s.rank + 60), 0) + COALESCE(1.0 / (d2.rank + 60), 0) AS rrf_score,
        CASE
            WHEN s.rank IS NOT NULL AND d2.rank IS NOT NULL THEN 'hybrid'
            WHEN s.rank IS NOT NULL THEN 'sparse'
            ELSE 'dense'
        END AS source_type
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    LEFT JOIN sparse s ON dc.id = s.id
    LEFT JOIN dense d2 ON dc.id = d2.id
    WHERE s.rank IS NOT NULL OR d2.rank IS NOT NULL
    ORDER BY rrf_score DESC
    LIMIT :rerank_top_k;
    """
    
    # 3. Execute the hybrid query
    result = await db.execute(text(hybrid_sql), {
        "query": query_text,
        "scope_id": scope_id,
        "query_vector": query_vector_str,
        "rerank_top_k": top_k * 2
    })

    # 4. Format retrieved chunks
    retrieved_chunks = []
    for row in result.mappings():
        retrieved_chunks.append({
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "filename": row["filename"],
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "score": float(row["rrf_score"]),
            "source_type": row["source_type"]
        })

    # 5. Rerank using CrossEncoder
    if retrieved_chunks:
        retrieved_chunks = await embedding_service.rerank_chunks(query_text, retrieved_chunks)

    # 6. Return Top K
    return retrieved_chunks[:top_k]