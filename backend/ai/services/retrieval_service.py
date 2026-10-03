import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.embedding_service import embedding_service, EmbeddingTask
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from app.config.settings import settings
import re
import logging

logger = logging.getLogger(__name__)

def build_sparse_query(text: str) -> str:
    tokens = re.findall(r'\w+', text)
    unique_tokens = []
    seen = set()
    for t in tokens:
        t_lower = t.lower()
        if t_lower and t_lower != 'or' and t_lower not in seen:
            seen.add(t_lower)
            unique_tokens.append(t_lower)
    return " or ".join(unique_tokens)


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

    # 2. HNSW Settings
    try:
        await db.execute(text("SAVEPOINT hnsw_sp"))
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {settings.HNSW_EF_SEARCH}"))
        if settings.HNSW_ITERATIVE_SCAN and settings.HNSW_ITERATIVE_SCAN.lower() != "off":
            await db.execute(text(f"SET LOCAL hnsw.iterative_scan = '{settings.HNSW_ITERATIVE_SCAN}'"))
        await db.execute(text("RELEASE SAVEPOINT hnsw_sp"))
    except Exception as e:
        await db.execute(text("ROLLBACK TO SAVEPOINT hnsw_sp"))
        logger.warning(f"Failed to set pgvector HNSW parameters: {e}")

    # 3. Hybrid SQL Query with RRF
    sparse_query = build_sparse_query(query_text)
    
    if sparse_query:
        sparse_cte = f"""
        sparse_inner AS (
            SELECT
                dc.id,
                ts_rank_cd(dc.fts_tokens, websearch_to_tsquery('english', :sparse_query)) as rank_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {scope_filter}
              AND dc.fts_tokens @@ websearch_to_tsquery('english', :sparse_query)
            ORDER BY rank_score DESC
            LIMIT :list_size
        ),
        sparse AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY rank_score DESC) as rank
            FROM sparse_inner
        )"""
    else:
        sparse_cte = """
        sparse AS (
            SELECT CAST(NULL AS uuid) as id, CAST(NULL AS bigint) as rank WHERE false
        )"""

    dense_cte = f"""
    dense_inner AS (
        SELECT
            dc.id,
            dc.embedding <=> CAST(:query_vector AS vector) AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE {scope_filter}
        ORDER BY distance
        LIMIT :list_size
    ),
    dense AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY distance) as rank
        FROM dense_inner
    )"""

    hybrid_sql = f"""
    WITH {sparse_cte},
    {dense_cte}
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.chunk_index,
        dc.page_number,
        dc.content,
        d.filename,
        COALESCE(1.0 / (s.rank + :rrf_k), 0) + COALESCE(1.0 / (d2.rank + :rrf_k), 0) AS rrf_score,
        s.rank AS sparse_rank,
        d2.rank AS dense_rank,
        1.0 - (dc.embedding <=> CAST(:query_vector AS vector)) AS dense_similarity,
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
    ORDER BY rrf_score DESC, chunk_id
    LIMIT :rerank_candidates;
    """
    
    # 4. Execute the hybrid query
    result = await db.execute(text(hybrid_sql), {
        "query": query_text,
        "sparse_query": sparse_query,
        "scope_id": scope_id,
        "query_vector": query_vector_str,
        "list_size": settings.RETRIEVAL_LIST_SIZE,
        "rrf_k": settings.RRF_K,
        "rerank_candidates": max(settings.RERANK_CANDIDATES, top_k)
    })

    # 5. Format retrieved chunks
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
            "source_type": row["source_type"],
            "sparse_rank": row["sparse_rank"],
            "dense_rank": row["dense_rank"],
            "dense_similarity": float(row["dense_similarity"]) if row["dense_similarity"] is not None else None,
        })
        
    print(f"\n🔍 [Hybrid Search] Retrieved {len(retrieved_chunks)} candidates before reranking.")
    for i, c in enumerate(retrieved_chunks[:3]):  # Log top 3
        print(f"   Candidate {i+1}: RRF={c['score']:.4f} | Source={c['source_type']} | Dense Rank={c['dense_rank']} | Sparse Rank={c['sparse_rank']} | File={c['filename']}")

    # 6. Rerank using CrossEncoder
    if retrieved_chunks:
        print(f"⚖️ [Reranking] Passing {len(retrieved_chunks)} candidates to Cross-Encoder...")
        retrieved_chunks = await embedding_service.rerank_chunks(query_text, retrieved_chunks)
        
        print(f"✅ [Reranking] Done. Top {top_k} results:")
        for i, c in enumerate(retrieved_chunks[:top_k]):
            print(f"   Rank {i+1}: Score={c.get('rerank_score', 0):.4f} | File={c['filename']} | Chunk Index={c['chunk_index']}")
    print("-" * 50)

    # 7. Return Top K
    return retrieved_chunks[:top_k]