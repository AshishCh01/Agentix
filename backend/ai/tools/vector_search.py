import logging
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Lightweight Cross-Encoder Reranker initialization
_reranker_model = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("✅ Cross-Encoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Cross-Encoder model load failed: {e}. Falling back to RRF scores.")
            _reranker_model = False
    return _reranker_model


async def vector_search_tool(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    query: str,
    image_data: Optional[str] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Performs Hybrid Search (Dense pgvector + Sparse TSVector full-text) with
    Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
    """
    from ai.services.embedding_service import embedding_service

    # Ensure session_id is a valid string for SQL bindings
    session_id_str = str(session_id)

    # 1. Generate Query Vector
    if hasattr(embedding_service, "get_embedding"):
        query_vector = await embedding_service.get_embedding(query)
    else:
        query_vector = embedding_service.generate_embedding(query)

    vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # 2. Hybrid Search SQL using explicit CAST(:vector_str AS vector)
    hybrid_query = text("""
        WITH dense_search AS (
            SELECT 
                dc.id,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                d.filename,
                1 - (dc.embedding <=> CAST(:vector_str AS vector)) AS dense_score,
                ROW_NUMBER() OVER (ORDER BY dc.embedding <=> CAST(:vector_str AS vector) ASC) AS dense_rank
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.session_id = :session_id
            LIMIT 20
        ),
        sparse_search AS (
            SELECT 
                dc.id,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                d.filename,
                ts_rank_cd(dc.fts_tokens, plainto_tsquery('english', :query)) AS sparse_score,
                ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.fts_tokens, plainto_tsquery('english', :query)) DESC) AS sparse_rank
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.session_id = :session_id AND dc.fts_tokens @@ plainto_tsquery('english', :query)
            LIMIT 20
        )
        SELECT 
            COALESCE(ds.id, ss.id) AS id,
            COALESCE(ds.content, ss.content) AS content,
            COALESCE(ds.page_number, ss.page_number) AS page_number,
            COALESCE(ds.chunk_index, ss.chunk_index) AS chunk_index,
            COALESCE(ds.filename, ss.filename) AS filename,
            COALESCE(ds.dense_score, 0.0) AS dense_score,
            -- Reciprocal Rank Fusion (RRF) formula: 1 / (60 + rank)
            (COALESCE(1.0 / (60 + ds.dense_rank), 0.0) + COALESCE(1.0 / (60 + ss.sparse_rank), 0.0)) AS rrf_score
        FROM dense_search ds
        FULL OUTER JOIN sparse_search ss ON ds.id = ss.id
        ORDER BY rrf_score DESC
        LIMIT 10;
    """)

    try:
        result = await db.execute(
            hybrid_query,
            {"session_id": session_id_str, "vector_str": vector_str, "query": query},
        )
        rows = result.fetchall()
    except Exception as e:
        logger.error(f"❌ Hybrid Search SQL execution error: {str(e)}")
        rows = []

    if not rows:
        return {"chunks": [], "context_text": ""}

    candidate_chunks = [
        {
            "id": str(r.id),
            "content": r.content,
            "page_number": r.page_number or 1,
            "chunk_index": r.chunk_index,
            "filename": r.filename,
            "score": float(r.rrf_score),
        }
        for r in rows
    ]

    # 3. Cross-Encoder Reranking Pass
    reranker = get_reranker()
    if reranker and candidate_chunks:
        try:
            pairs = [[query, chunk["content"]] for chunk in candidate_chunks]
            scores = reranker.predict(pairs)
            for idx, score in enumerate(scores):
                candidate_chunks[idx]["rerank_score"] = float(score)

            candidate_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed, using default RRF order: {e}")

    # Top-K final context selection
    top_chunks = candidate_chunks[:top_k]

    # Format context block for LLM prompt
    context_blocks = []
    for c in top_chunks:
        context_blocks.append(
            f"[Source: {c['filename']} | Page {c['page_number']}]\n{c['content']}"
        )

    formatted_context = "\n\n---\n\n".join(context_blocks)

    return {
        "chunks": top_chunks,
        "context_text": formatted_context,
    }