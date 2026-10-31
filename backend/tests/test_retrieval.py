import os
import re
import uuid
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.main import app
from app.config.settings import settings
from app.models.base import Base
from app.models.user import User
from app.models.session import ChatSession
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from ai.services.retrieval_service import build_sparse_query, search_similar_chunks
from ai.services.embedding_service import embedding_service

def test_build_sparse_query():
    assert build_sparse_query("hello world") == "hello or world"
    assert build_sparse_query("hello OR world") == "hello or world"
    assert build_sparse_query("Hello WORLD") == "hello or world"
    assert build_sparse_query("a, b, c!") == "a or b or c"
    assert build_sparse_query("or or or") == ""
    assert build_sparse_query("!?") == ""
    assert build_sparse_query("hello hello world") == "hello or world"
    assert build_sparse_query("こんにちは world") == "こんにちは or world"

@pytest.mark.asyncio
async def test_search_endpoint_empty_session():
    """
    Verifies that a search on a non-existent or empty session gracefully returns an empty list
    without throwing a 500 retrieval error.
    """
    # Mock authentication to allow request
    app.dependency_overrides = {}
    from app.auth.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"id": "12345", "sub": "12345"}

    # Mock the database to return an empty session to simulate access denied / not found
    from app.database import crud
    
    with patch("app.database.crud.get_chat_session", new_callable=AsyncMock) as mock_get_session:
        # If session not found, it throws a 404
        mock_get_session.return_value = None
        
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search",
                json={
                    "session_id": "00000000-0000-0000-0000-000000000000",
                    "query": "test query",
                    "top_k": 4
                }
            )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_search_endpoint_success():
    """
    Verifies the hybrid retrieval pipeline returns the expected schema properties:
    score (not similarity_score), source_type, page_number, filename.
    """
    app.dependency_overrides = {}
    from app.auth.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"id": "12345"}

    from app.database import crud
    
    with patch("app.database.crud.get_chat_session", new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = {"id": "session-123"}
        
        with patch("app.routes.search.search_similar_chunks", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "filename": "test.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "test content",
                    "score": 0.95,
                    "source_type": "hybrid",
                    "rerank_score": 0.99,
                    "rerank_logit": 3.5
                }
            ]
            
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/search",
                    json={
                        "session_id": "11111111-1111-1111-1111-111111111111",
                        "query": "test query",
                        "top_k": 4
                    }
                )
                
            if response.status_code != 200:
                print(response.json())
            
            assert response.status_code == 200
            data = response.json()
            assert data["results_count"] == 1
            chunk = data["chunks"][0]
            
            # Verify exact schema mapping
            assert chunk["score"] == 0.95
            assert "similarity_score" not in chunk
            assert chunk["source_type"] == "hybrid"  # Changed from 'reranked' as the new reranker leaves source_type unchanged
            # rerank_score is added internally but stripped by the public ChunkResult schema.
            # We will test the internal rerank_score in a separate test below.
            assert chunk["page_number"] == 1
            assert chunk["filename"] == "test.pdf"

@pytest.mark.asyncio
async def test_search_endpoint_empty_results():
    """
    Verifies that a successful search returning no chunks genuinely returns 0 results
    and not an error.
    """
    app.dependency_overrides = {}
    from app.auth.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: {"id": "12345"}

    with patch("app.database.crud.get_chat_session", new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = {"id": "session-123"}
        
        with patch("app.routes.search.search_similar_chunks", new_callable=AsyncMock) as mock_search:
            # Simulate DB returning empty lists natively
            mock_search.return_value = []
            
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/search",
                    json={
                        "session_id": "11111111-1111-1111-1111-111111111111",
                        "query": "test query"
                    }
                )
                
            assert response.status_code == 200
            data = response.json()
            assert data["results_count"] == 0
            assert data["chunks"] == []

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_search_endpoint_success())

@pytest.mark.asyncio
async def test_rerank_chunks_success():
    from ai.services.embedding_service import embedding_service
    from app.config.settings import settings
    
    settings.RERANK_PROVIDER = "local"
    
    # Mock reranker predict
    class MockReranker:
        def predict(self, inputs, batch_size=8, show_progress_bar=False):
            return [0.5, 2.5]
            
    embedding_service._reranker = MockReranker()
    embedding_service._reranker_unavailable = False
    
    chunks = [
        {"content": "bad chunk", "score": 0.8, "source_type": "hybrid"},
        {"content": "good chunk", "score": 0.6, "source_type": "dense"}
    ]
    
    result = await embedding_service.rerank_chunks("query", chunks)
    
    assert len(result) == 2
    assert result[0]["content"] == "good chunk"  # Reordered because 2.5 > 0.5
    assert result[0]["source_type"] == "dense"   # Unchanged
    assert "rerank_score" in result[0]
    assert 0 <= result[0]["rerank_score"] <= 1
    assert "rerank_logit" in result[0]
    assert result[0]["rerank_logit"] == 2.5

@pytest.mark.asyncio
async def test_rerank_chunks_predict_raises():
    from ai.services.embedding_service import embedding_service
    from app.config.settings import settings
    
    settings.RERANK_PROVIDER = "local"
    
    class MockRerankerError:
        def predict(self, inputs, batch_size=8, show_progress_bar=False):
            raise ValueError("Predict failed")
            
    embedding_service._reranker = MockRerankerError()
    embedding_service._reranker_unavailable = False
    
    chunks = [
        {"content": "chunk1", "score": 0.8, "source_type": "hybrid"},
        {"content": "chunk2", "score": 0.6, "source_type": "dense"}
    ]
    
    result = await embedding_service.rerank_chunks("query", list(chunks))
    
    # Order unchanged
    assert result[0]["content"] == "chunk1"
    assert result[1]["content"] == "chunk2"
    assert "rerank_score" not in result[0]

@pytest.mark.asyncio
async def test_rerank_chunks_provider_none():
    from ai.services.embedding_service import embedding_service
    from app.config.settings import settings
    
    settings.RERANK_PROVIDER = "none"
    embedding_service._reranker = None
    embedding_service._reranker_unavailable = False
    
    chunks = [{"content": "chunk1", "score": 0.8, "source_type": "hybrid"}]
    result = await embedding_service.rerank_chunks("query", list(chunks))
    assert "rerank_score" not in result[0]
    assert embedding_service._reranker is None

@pytest.mark.asyncio
async def test_reranker_quantize_fallback():
    from ai.services.embedding_service import embedding_service
    from app.config.settings import settings
    import torch
    
    settings.RERANK_PROVIDER = "local"
    settings.RERANK_QUANTIZE = True
    embedding_service._reranker = None
    embedding_service._reranker_unavailable = False
    
    with patch("torch.quantization.quantize_dynamic", side_effect=ValueError("Quantize error")):
        embedding_service.load_reranker()
        
    assert embedding_service._reranker is not None
    assert not embedding_service._reranker_unavailable

def test_routing_retrieval_error():
    from ai.agents.graph import route_after_vector_search
    state = {"retrieved_chunks": [], "retrieval_error": True}
    assert route_after_vector_search(state) == "answer"

def test_routing_empty_chunks():
    from ai.agents.graph import route_after_vector_search
    state = {"retrieved_chunks": [], "retrieval_error": False}
    assert route_after_vector_search(state) == "web_search"

@pytest.mark.asyncio
async def test_vector_search_db_error():
    from ai.tools.vector_search import vector_search_tool
    import uuid
    
    with patch("ai.tools.vector_search.search_similar_chunks", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = Exception("DB Connection Error")
        result = await vector_search_tool(db=None, session_id=uuid.uuid4(), query="test")
        
        assert result["chunks"] == []
        assert "retrieval_error" in result
        assert "DB Connection Error" in result["retrieval_error"]




# =====================================================================
# Issue #30: Hybrid SQL Rewrite Tests
# =====================================================================

class _MockDBSession:
    """
    Mock AsyncSession that captures executed statements and parameters,
    simulates SET LOCAL failures, and returns mock result mappings.
    """
    def __init__(self, rows=None, fail_on_set_local=False, set_local_exc=None):
        self.rows = rows or []
        self.fail_on_set_local = fail_on_set_local
        self.set_local_exc = set_local_exc or Exception("simulated SET LOCAL error")
        self.executed_queries = []
        self.executed_params = []

    async def execute(self, statement, params=None):
        sql_text = statement.text if hasattr(statement, "text") else str(statement)
        self.executed_queries.append(sql_text)
        self.executed_params.append(params)

        if "SET LOCAL" in sql_text and self.fail_on_set_local:
            raise self.set_local_exc

        mock_result = MagicMock()
        mock_result.mappings.return_value = list(self.rows)
        return mock_result


def _build_mock_row(
    chunk_id: str,
    doc_id: str = "doc-1",
    filename: str = "test.pdf",
    chunk_index: int = 0,
    page_number: int = 1,
    content: str = "sample text",
    rrf_score: float = 0.05,
    source_type: str = "hybrid",
    sparse_rank: Optional[int] = 1,
    dense_rank: Optional[int] = 1,
    dense_similarity: Optional[float] = 0.9,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content,
        "rrf_score": rrf_score,
        "source_type": source_type,
        "sparse_rank": sparse_rank,
        "dense_rank": dense_rank,
        "dense_similarity": dense_similarity,
    }


# ---------------------------------------------------------------------
# A. SQL Structure and LIMIT Placement Tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_sql_dense_branch_structure_and_limit():
    """
    Verify that the dense retrieval branch uses the correct SQL structure:
    - dense_inner subquery contains ORDER BY distance
    - dense_inner subquery contains LIMIT :list_size
    - ORDER BY distance appears strictly BEFORE LIMIT :list_size inside dense_inner
    - dense CTE computes ROW_NUMBER() OVER (ORDER BY distance)
    - LIMIT is placed inside dense_inner and not outside
    """
    db = _MockDBSession()
    session_id = uuid.uuid4()

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(db, query_text="machine learning", session_id=session_id)

    hybrid_sql = next(q for q in db.executed_queries if "dense_inner" in q)

    dense_inner_match = re.search(r"dense_inner\s+AS\s*\((.*?)\)\s*,\s*dense\s+AS", hybrid_sql, re.DOTALL | re.IGNORECASE)
    assert dense_inner_match is not None, "dense_inner CTE subquery not found in hybrid SQL"
    dense_inner_sql = dense_inner_match.group(1)

    assert "ORDER BY distance" in dense_inner_sql, "ORDER BY distance missing from dense_inner subquery"
    assert "LIMIT :list_size" in dense_inner_sql, "LIMIT :list_size missing from dense_inner subquery"
    order_idx = dense_inner_sql.index("ORDER BY distance")
    limit_idx = dense_inner_sql.index("LIMIT :list_size")
    assert order_idx < limit_idx, "ORDER BY distance must precede LIMIT inside dense_inner"

    assert "dense AS (" in hybrid_sql, "dense CTE not found in hybrid SQL"
    dense_part = hybrid_sql.split("dense AS (", 1)[1].split("FROM dense_inner", 1)[0]
    assert "ROW_NUMBER() OVER (ORDER BY distance)" in dense_part
    dense_full = hybrid_sql.split("dense AS (", 1)[1].split("FROM dense_inner", 1)[1].split(")", 1)[0]
    assert "LIMIT" not in dense_full.upper(), "LIMIT must not be in outer dense CTE"


@pytest.mark.asyncio
async def test_hybrid_sql_sparse_branch_structure_and_limit():
    """
    Verify that the sparse retrieval branch uses the correct SQL structure:
    - sparse_inner subquery contains ORDER BY rank_score DESC
    - sparse_inner subquery contains LIMIT :list_size
    - ORDER BY rank_score DESC appears strictly BEFORE LIMIT :list_size inside sparse_inner
    - sparse CTE computes ROW_NUMBER() OVER (ORDER BY rank_score DESC)
    - LIMIT is placed inside sparse_inner and not outside
    """
    db = _MockDBSession()
    session_id = uuid.uuid4()

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(db, query_text="natural language processing search", session_id=session_id)

    hybrid_sql = next(q for q in db.executed_queries if "sparse_inner" in q)

    sparse_inner_match = re.search(r"sparse_inner\s+AS\s*\((.*?)\)\s*,\s*sparse\s+AS", hybrid_sql, re.DOTALL | re.IGNORECASE)
    assert sparse_inner_match is not None, "sparse_inner CTE subquery not found in hybrid SQL"
    sparse_inner_sql = sparse_inner_match.group(1)

    assert "ORDER BY rank_score DESC" in sparse_inner_sql, "ORDER BY rank_score DESC missing from sparse_inner subquery"
    assert "LIMIT :list_size" in sparse_inner_sql, "LIMIT :list_size missing from sparse_inner subquery"
    order_idx = sparse_inner_sql.index("ORDER BY rank_score DESC")
    limit_idx = sparse_inner_sql.index("LIMIT :list_size")
    assert order_idx < limit_idx, "ORDER BY rank_score DESC must precede LIMIT inside sparse_inner"

    assert "sparse AS (" in hybrid_sql, "sparse CTE not found in hybrid SQL"
    sparse_part = hybrid_sql.split("sparse AS (", 1)[1].split("FROM sparse_inner", 1)[0]
    assert "ROW_NUMBER() OVER (ORDER BY rank_score DESC)" in sparse_part
    sparse_full = hybrid_sql.split("sparse AS (", 1)[1].split("FROM sparse_inner", 1)[1].split(")", 1)[0]
    assert "LIMIT" not in sparse_full.upper(), "LIMIT must not be in outer sparse CTE"


@pytest.mark.asyncio
async def test_hybrid_sql_sparse_branch_fallback_on_empty_query():
    """
    Verify that when query produces an empty sparse query (e.g. symbols or 'or or or'):
    - sparse_inner is omitted
    - sparse CTE falls back to empty dummy query (WHERE false)
    - dense branch still maintains its ORDER BY distance and LIMIT :list_size
    """
    db = _MockDBSession()
    session_id = uuid.uuid4()

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(db, query_text="??? !!! ...", session_id=session_id)

    hybrid_sql = next(q for q in db.executed_queries if "SELECT" in q and "document_chunks" in q)

    assert "sparse_inner" not in hybrid_sql
    assert "sparse AS (" in hybrid_sql
    assert "WHERE false" in hybrid_sql

    # Dense inner must still be present with correct order and limit
    dense_inner_match = re.search(r"dense_inner\s+AS\s*\((.*?)\)\s*,\s*dense\s+AS", hybrid_sql, re.DOTALL | re.IGNORECASE)
    assert dense_inner_match is not None
    dense_inner_sql = dense_inner_match.group(1)
    order_pos = dense_inner_sql.find("ORDER BY distance")
    limit_pos = dense_inner_sql.find("LIMIT :list_size")
    assert order_pos != -1 and limit_pos != -1 and order_pos < limit_pos


@pytest.mark.asyncio
async def test_hybrid_sql_outer_query_deterministic_candidate_selection():
    """
    Verify that the outer query preserves deterministic candidate selection:
    - Outer query has ORDER BY rrf_score DESC, chunk_id (tie-breaker)
    - ORDER BY rrf_score DESC, chunk_id appears strictly BEFORE LIMIT :rerank_candidates
    - Required parameters (list_size, rrf_k, rerank_candidates, scope_id) are bound correctly
    """
    db = _MockDBSession()
    session_id = uuid.uuid4()

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(db, query_text="cloud infrastructure", session_id=session_id, top_k=7)

    query_idx = next(i for i, q in enumerate(db.executed_queries) if "rrf_score DESC" in q)
    hybrid_sql = db.executed_queries[query_idx]
    params = db.executed_params[query_idx]

    # Verify deterministic tie-breaker in outer ORDER BY
    assert "ORDER BY rrf_score DESC, chunk_id" in hybrid_sql, "ORDER BY rrf_score DESC, chunk_id missing from outer query"
    assert "LIMIT :rerank_candidates" in hybrid_sql, "LIMIT :rerank_candidates missing from outer query"
    order_idx = hybrid_sql.index("ORDER BY rrf_score DESC, chunk_id")
    limit_idx = hybrid_sql.index("LIMIT :rerank_candidates")
    assert order_idx < limit_idx, "Outer ORDER BY must precede LIMIT :rerank_candidates"

    # Verify bound parameters
    assert params["list_size"] == settings.RETRIEVAL_LIST_SIZE
    assert params["rrf_k"] == settings.RRF_K
    assert params["rerank_candidates"] == max(settings.RERANK_CANDIDATES, 7)
    assert params["scope_id"] == str(session_id)
    assert "query_vector" in params
    assert params["sparse_query"] == "cloud or infrastructure"


@pytest.mark.asyncio
async def test_hybrid_sql_scope_filter_subquery_placement():
    """
    Verify scope filtering is placed inside both sparse_inner and dense_inner:
    - Session scope: d.session_id = :scope_id in both subqueries
    - KB scope: d.knowledge_base_id = :scope_id in both subqueries
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS

        # 1. Session scoped
        db_session = _MockDBSession()
        await search_similar_chunks(db_session, query_text="vector database", session_id=session_id, knowledge_base_id=None)
        sql_session = next(q for q in db_session.executed_queries if "dense_inner" in q)
        assert "d.session_id = :scope_id" in sql_session
        assert "d.knowledge_base_id" not in sql_session

        # 2. Knowledge Base scoped
        db_kb = _MockDBSession()
        await search_similar_chunks(db_kb, query_text="vector database", session_id=session_id, knowledge_base_id=kb_id)
        sql_kb = next(q for q in db_kb.executed_queries if "dense_inner" in q)
        assert "d.knowledge_base_id = :scope_id" in sql_kb
        assert "d.session_id = :scope_id" not in sql_kb


# ---------------------------------------------------------------------
# B. Correct Number of Returned Candidates
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidate_count_scenario_1_reranker_succeeds(monkeypatch):
    """
    Scenario 1: Reranker succeeds
    When sufficient candidates are available and reranking works:
    - Exactly top_k candidates are returned.
    - Candidates are reordered according to CrossEncoder rerank_score.
    """
    rows = [
        _build_mock_row("c1", content="chunk 1", rrf_score=0.9),
        _build_mock_row("c2", content="chunk 2", rrf_score=0.8),
        _build_mock_row("c3", content="chunk 3", rrf_score=0.7),
        _build_mock_row("c4", content="chunk 4", rrf_score=0.6),
        _build_mock_row("c5", content="chunk 5", rrf_score=0.5),
        _build_mock_row("c6", content="chunk 6", rrf_score=0.4),
    ]
    db = _MockDBSession(rows=rows)
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "local")

    class _MockSuccessReranker:
        def predict(self, inputs, batch_size=8, show_progress_bar=False):
            # Assign highest logit to c4, then c2, then c5, then c1, c3, c6
            return [-1.0, 3.0, 0.5, 6.0, 2.0, -2.0]

    monkeypatch.setattr(embedding_service, "_reranker", _MockSuccessReranker())
    monkeypatch.setattr(embedding_service, "_reranker_unavailable", False)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        results = await search_similar_chunks(db, query_text="test", session_id=uuid.uuid4(), top_k=3)

    assert len(results) == 3, f"Expected exactly top_k=3 candidates, got {len(results)}"
    assert results[0]["chunk_id"] == "c4"
    assert results[1]["chunk_id"] == "c2"
    assert results[2]["chunk_id"] == "c5"
    assert "rerank_score" in results[0]
    assert results[0]["rerank_score"] > results[1]["rerank_score"] > results[2]["rerank_score"]


@pytest.mark.asyncio
async def test_candidate_count_scenario_2_reranker_fails_open(monkeypatch):
    """
    Scenario 2: Reranker fails open
    When reranking raises an exception:
    - Retrieval does not fail entirely (no exception raised).
    - Fallback behavior returns expected candidates.
    - Exactly top_k candidates are returned when enough exist.
    - Fallback preserves the intended pre-reranked RRF ranking/order.
    """
    rows = [
        _build_mock_row("c1", content="chunk 1", rrf_score=0.9),
        _build_mock_row("c2", content="chunk 2", rrf_score=0.8),
        _build_mock_row("c3", content="chunk 3", rrf_score=0.7),
        _build_mock_row("c4", content="chunk 4", rrf_score=0.6),
        _build_mock_row("c5", content="chunk 5", rrf_score=0.5),
        _build_mock_row("c6", content="chunk 6", rrf_score=0.4),
    ]
    db = _MockDBSession(rows=rows)
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "local")

    class _MockFailingReranker:
        def predict(self, inputs, batch_size=8, show_progress_bar=False):
            raise RuntimeError("CrossEncoder model execution failed")

    monkeypatch.setattr(embedding_service, "_reranker", _MockFailingReranker())
    monkeypatch.setattr(embedding_service, "_reranker_unavailable", False)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        results = await search_similar_chunks(db, query_text="test", session_id=uuid.uuid4(), top_k=4)

    assert len(results) == 4, f"Expected exactly top_k=4 candidates, got {len(results)}"
    assert [r["chunk_id"] for r in results] == ["c1", "c2", "c3", "c4"], "Intended RRF fallback ordering must be preserved"
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.8
    assert results[2]["score"] == 0.7
    assert results[3]["score"] == 0.6


@pytest.mark.asyncio
async def test_candidate_count_scenario_3_reranking_disabled(monkeypatch):
    """
    Scenario 3: Reranking disabled (RERANK_PROVIDER=none)
    - No reranker execution is required.
    - Hybrid retrieval still returns results.
    - Exactly top_k candidates are returned when sufficient candidates exist.
    - The intended non-reranked ordering is preserved.
    """
    rows = [
        _build_mock_row("c1", content="chunk 1", rrf_score=0.95),
        _build_mock_row("c2", content="chunk 2", rrf_score=0.85),
        _build_mock_row("c3", content="chunk 3", rrf_score=0.75),
        _build_mock_row("c4", content="chunk 4", rrf_score=0.65),
    ]
    db = _MockDBSession(rows=rows)
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)
    monkeypatch.setattr(embedding_service, "_reranker_unavailable", False)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        results = await search_similar_chunks(db, query_text="test", session_id=uuid.uuid4(), top_k=3)

    assert len(results) == 3, f"Expected exactly top_k=3 candidates, got {len(results)}"
    assert [r["chunk_id"] for r in results] == ["c1", "c2", "c3"]
    assert "rerank_score" not in results[0]
    assert results[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_candidate_count_scenario_4_insufficient_candidates(monkeypatch):
    """
    Scenario 4: Insufficient candidates
    When fewer candidates than top_k are available:
    - All available candidates are returned.
    - No artificial candidates are created.
    - Result count is not incorrectly padded to top_k.
    - No exception occurs.
    """
    rows = [
        _build_mock_row("c1", content="single chunk", rrf_score=0.9),
        _build_mock_row("c2", content="second chunk", rrf_score=0.8),
    ]
    db = _MockDBSession(rows=rows)
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        results = await search_similar_chunks(db, query_text="test", session_id=uuid.uuid4(), top_k=5)

    assert len(results) == 2, f"Expected exactly 2 available candidates, got {len(results)}"
    assert [r["chunk_id"] for r in results] == ["c1", "c2"]
    assert all(r is not None and r["chunk_id"] in ("c1", "c2") for r in results)


# ---------------------------------------------------------------------
# C. SET LOCAL Failure Handling Tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_local_failure_handling_and_transaction_usability(caplog, monkeypatch):
    """
    Verify SET LOCAL failure handling:
    1. Retrieval continues and returns candidates.
    2. Exactly one warning is logged for the failed setting operation.
    3. The exception does not unnecessarily abort the retrieval flow.
    4. Subsequent database operations can still execute within the transaction.
    """
    rows = [
        _build_mock_row("c1", content="chunk 1", rrf_score=0.88),
        _build_mock_row("c2", content="chunk 2", rrf_score=0.77),
    ]
    set_local_error = Exception("unrecognized configuration parameter 'hnsw.ef_search'")
    db = _MockDBSession(rows=rows, fail_on_set_local=True, set_local_exc=set_local_error)
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS

        with caplog.at_level(logging.WARNING):
            results = await search_similar_chunks(db, query_text="test query", session_id=uuid.uuid4(), top_k=2)

    # 1. Retrieval continues and returns candidates
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"

    # 2. Exactly one warning logged for the failed setting operation
    hnsw_warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "Failed to set pgvector HNSW parameters" in r.message
    ]
    assert len(hnsw_warnings) == 1, f"Expected exactly 1 HNSW warning, found {len(hnsw_warnings)}"
    assert "hnsw.ef_search" in hnsw_warnings[0].message

    # 3. Verify sequence of queries (SAVEPOINT -> SET LOCAL (fails) -> ROLLBACK TO SAVEPOINT -> hybrid_sql)
    queries = db.executed_queries
    assert any("SAVEPOINT hnsw_sp" in q for q in queries)
    assert any("ROLLBACK TO SAVEPOINT hnsw_sp" in q for q in queries)
    assert any("document_chunks" in q for q in queries)

    # 4. Subsequent database operations can still execute within the transaction
    subsequent_res = await db.execute(text("SELECT 1;"))
    assert subsequent_res is not None
    assert any("SELECT 1;" in q for q in db.executed_queries)


# ---------------------------------------------------------------------
# D. PostgreSQL Integration Tests
# ---------------------------------------------------------------------

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
requires_test_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL environment variable is not configured. Skipping PostgreSQL integration tests.",
)


@pytest.fixture
async def integration_db():
    """
    Provides a real PostgreSQL session with transaction rollback cleanup.
    Ensures tests are completely isolated, repeatable, and never leave artifacts.
    """
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not configured")

    if "supabase.co" in TEST_DATABASE_URL:
        pytest.skip("Safety guard: TEST_DATABASE_URL points to Supabase. Integration tests must not run against production.")

    test_async_url = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql+asyncpg://", TEST_DATABASE_URL)
    engine = create_async_engine(test_async_url, echo=False)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()

    await engine.dispose()


async def _create_test_document_and_chunks(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_id: Optional[uuid.UUID],
    knowledge_base_id: Optional[uuid.UUID],
    filename: str,
    chunks_data: List[dict],
) -> Document:
    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        file_type="application/pdf",
        file_path=f"/test/{filename}",
        file_size=1024,
        status="completed",
    )
    db.add(doc)
    await db.flush()

    for idx, c in enumerate(chunks_data):
        chunk = DocumentChunk(
            id=c.get("id") or uuid.uuid4(),
            document_id=doc.id,
            chunk_index=idx,
            page_number=1,
            content=c["content"],
            embedding=c["embedding"],
        )
        db.add(chunk)

    await db.flush()
    return doc


@requires_test_database
@pytest.mark.asyncio
async def test_integration_dense_nearest_neighbor(integration_db: AsyncSession, monkeypatch):
    """
    1. Dense nearest-neighbor retrieval integration test:
    - Insert deterministic test vectors.
    - Execute actual dense retrieval.
    - Verify that the true nearest-neighbor chunk appears in the dense retrieval list.
    """
    dims = settings.EMBEDDING_DIMENSIONS
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    user = User(id=user_id, email=f"user_{user_id.hex[:6]}@example.com")
    session = ChatSession(id=session_id, user_id=user_id, title="Integration Test Session")
    integration_db.add_all([user, session])
    await integration_db.flush()

    # Deterministic vectors
    q_vec = [1.0] + [0.0] * (dims - 1)
    c1_vec = [1.0] + [0.0] * (dims - 1)           # Distance = 0.0 (exact match)
    c2_vec = [0.0, 1.0] + [0.0] * (dims - 2)      # Distance = 1.0
    c3_vec = [-1.0] + [0.0] * (dims - 1)          # Distance = 2.0 (opposite)

    c1_id = uuid.uuid4()
    chunks_data = [
        {"id": c1_id, "content": "Nearest neighbor chunk content", "embedding": c1_vec},
        {"id": uuid.uuid4(), "content": "Orthogonal chunk content", "embedding": c2_vec},
        {"id": uuid.uuid4(), "content": "Opposite chunk content", "embedding": c3_vec},
    ]
    await _create_test_document_and_chunks(integration_db, user_id, session_id, None, "dense_test.pdf", chunks_data)

    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = q_vec
        results = await search_similar_chunks(integration_db, query_text="query", session_id=session_id, top_k=3)

    assert len(results) >= 1
    # True nearest neighbor must have dense_rank = 1 and appears first
    assert results[0]["chunk_id"] == str(c1_id)
    assert results[0]["dense_rank"] == 1
    assert results[0]["dense_similarity"] is not None
    assert abs(results[0]["dense_similarity"] - 1.0) < 1e-4


@requires_test_database
@pytest.mark.asyncio
async def test_integration_sparse_multi_word_retrieval(integration_db: AsyncSession, monkeypatch):
    """
    2. Sparse multi-word retrieval integration test:
    - Prepare indexed text containing known searchable words.
    - Execute an actual multi-word search query.
    - Verify that the sparse retrieval list is non-empty.
    - Confirm that expected matching content appears.
    """
    dims = settings.EMBEDDING_DIMENSIONS
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    user = User(id=user_id, email=f"user_{user_id.hex[:6]}@example.com")
    session = ChatSession(id=session_id, user_id=user_id, title="Integration Test Session")
    integration_db.add_all([user, session])
    await integration_db.flush()

    orthogonal_vec = [0.0] * dims
    target_id = uuid.uuid4()
    chunks_data = [
        {
            "id": target_id,
            "content": "PostgreSQL inverted full-text search indexing with reciprocal rank fusion",
            "embedding": orthogonal_vec,
        },
        {
            "id": uuid.uuid4(),
            "content": "Cellular respiration photosynthesis and botanical taxonomy",
            "embedding": orthogonal_vec,
        },
    ]
    await _create_test_document_and_chunks(integration_db, user_id, session_id, None, "sparse_test.pdf", chunks_data)

    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = orthogonal_vec
        results = await search_similar_chunks(
            integration_db,
            query_text="postgresql inverted reciprocal",
            session_id=session_id,
            top_k=2,
        )

    assert len(results) >= 1
    matching_chunk = next((r for r in results if r["chunk_id"] == str(target_id)), None)
    assert matching_chunk is not None, "Target matching chunk must appear in sparse retrieval results"
    assert matching_chunk["sparse_rank"] is not None
    assert "postgresql" in matching_chunk["content"].lower()


@requires_test_database
@pytest.mark.asyncio
async def test_integration_scope_isolation(integration_db: AsyncSession, monkeypatch):
    """
    3. Knowledge-base and document scope isolation integration test:
    - Prepare data belonging to separate knowledge bases and sessions.
    - Execute retrieval with a specified scope.
    - Verify that results do not contain chunks outside the requested scope.
    """
    dims = settings.EMBEDDING_DIMENSIONS
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user_{user_id.hex[:6]}@example.com")
    integration_db.add(user)

    # 1. Knowledge Base scope test
    kb1 = KnowledgeBase(id=uuid.uuid4(), user_id=user_id, name="KB Alpha")
    kb2 = KnowledgeBase(id=uuid.uuid4(), user_id=user_id, name="KB Beta")
    integration_db.add_all([kb1, kb2])
    await integration_db.flush()

    dummy_vec = [0.1] * dims
    kb1_chunk_id = uuid.uuid4()
    kb2_chunk_id = uuid.uuid4()

    await _create_test_document_and_chunks(
        integration_db, user_id, None, kb1.id, "kb1_doc.pdf",
        [{"id": kb1_chunk_id, "content": "Confidential roadmap for Project Alpha", "embedding": dummy_vec}],
    )
    await _create_test_document_and_chunks(
        integration_db, user_id, None, kb2.id, "kb2_doc.pdf",
        [{"id": kb2_chunk_id, "content": "Public guidelines for Project Beta", "embedding": dummy_vec}],
    )

    dummy_session = uuid.uuid4()
    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = dummy_vec
        results_kb1 = await search_similar_chunks(
            integration_db,
            query_text="roadmap guidelines",
            session_id=dummy_session,
            knowledge_base_id=kb1.id,
            top_k=5,
        )

    retrieved_ids_kb1 = [r["chunk_id"] for r in results_kb1]
    assert str(kb1_chunk_id) in retrieved_ids_kb1
    assert str(kb2_chunk_id) not in retrieved_ids_kb1, "Chunk from KB Beta leaked into KB Alpha query results"

    # 2. Chat Session scope test
    session1 = ChatSession(id=uuid.uuid4(), user_id=user_id, title="Session 1")
    session2 = ChatSession(id=uuid.uuid4(), user_id=user_id, title="Session 2")
    integration_db.add_all([session1, session2])
    await integration_db.flush()

    s1_chunk_id = uuid.uuid4()
    s2_chunk_id = uuid.uuid4()

    await _create_test_document_and_chunks(
        integration_db, user_id, session1.id, None, "s1_doc.pdf",
        [{"id": s1_chunk_id, "content": "Private session 1 notes", "embedding": dummy_vec}],
    )
    await _create_test_document_and_chunks(
        integration_db, user_id, session2.id, None, "s2_doc.pdf",
        [{"id": s2_chunk_id, "content": "Private session 2 notes", "embedding": dummy_vec}],
    )

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = dummy_vec
        results_s1 = await search_similar_chunks(
            integration_db,
            query_text="private notes",
            session_id=session1.id,
            top_k=5,
        )

    retrieved_ids_s1 = [r["chunk_id"] for r in results_s1]
    assert str(s1_chunk_id) in retrieved_ids_s1
    assert str(s2_chunk_id) not in retrieved_ids_s1, "Chunk from Session 2 leaked into Session 1 query results"


@requires_test_database
@pytest.mark.asyncio
async def test_integration_deterministic_ordering_with_ties(integration_db: AsyncSession, monkeypatch):
    """
    4. Deterministic ordering integration test:
    - Execute the same retrieval query repeatedly against unchanged test data.
    - Verify deterministic result ordering across multiple executions.
    - Include tied records to verify secondary ordering criterion (chunk_id).
    """
    dims = settings.EMBEDDING_DIMENSIONS
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    user = User(id=user_id, email=f"user_{user_id.hex[:6]}@example.com")
    session = ChatSession(id=session_id, user_id=user_id, title="Tie Ordering Session")
    integration_db.add_all([user, session])
    await integration_db.flush()

    # Create 4 chunks where chunks have identical content and identical embeddings (tied RRF score)
    identical_vec = [0.05] * dims
    chunks_data = [
        {"id": uuid.uuid4(), "content": "Deterministic benchmark identical text content", "embedding": identical_vec},
        {"id": uuid.uuid4(), "content": "Deterministic benchmark identical text content", "embedding": identical_vec},
        {"id": uuid.uuid4(), "content": "Deterministic benchmark identical text content", "embedding": identical_vec},
        {"id": uuid.uuid4(), "content": "Deterministic benchmark identical text content", "embedding": identical_vec},
    ]
    await _create_test_document_and_chunks(integration_db, user_id, session_id, None, "ties_doc.pdf", chunks_data)

    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = identical_vec

        run_orders = []
        for _ in range(5):
            results = await search_similar_chunks(
                integration_db,
                query_text="deterministic benchmark",
                session_id=session_id,
                top_k=4,
            )
            run_orders.append([r["chunk_id"] for r in results])

    assert len(run_orders) == 5
    first_order = run_orders[0]
    for i, order in enumerate(run_orders[1:], start=2):
        assert order == first_order, f"Run {i} returned order {order} which differs from initial order {first_order}"


@requires_test_database
@pytest.mark.asyncio
async def test_integration_set_local_failure_recovery(integration_db: AsyncSession, monkeypatch, caplog):
    """
    Verify SET LOCAL failure recovery against real PostgreSQL:
    - Attempting invalid HNSW parameter triggers SET LOCAL error.
    - SAVEPOINT rollback prevents transaction abort.
    - Exactly one warning logged.
    - Retrieval succeeds and subsequent SELECT 1 can execute within transaction.
    """
    dims = settings.EMBEDDING_DIMENSIONS
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    user = User(id=user_id, email=f"user_{user_id.hex[:6]}@example.com")
    session = ChatSession(id=session_id, user_id=user_id, title="Set Local Test Session")
    integration_db.add_all([user, session])
    await integration_db.flush()

    dummy_vec = [0.01] * dims
    c_id = uuid.uuid4()
    await _create_test_document_and_chunks(
        integration_db, user_id, session_id, None, "set_local_doc.pdf",
        [{"id": c_id, "content": "Sample content for set local recovery", "embedding": dummy_vec}],
    )

    monkeypatch.setattr(settings, "RERANK_PROVIDER", "none")
    monkeypatch.setattr(embedding_service, "_reranker", None)
    # Set invalid ef_search value to trigger PostgreSQL SET LOCAL failure
    monkeypatch.setattr(settings, "HNSW_EF_SEARCH", -1)

    with patch.object(embedding_service, "generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = dummy_vec
        with caplog.at_level(logging.WARNING):
            results = await search_similar_chunks(
                integration_db,
                query_text="sample content",
                session_id=session_id,
                top_k=2,
            )

    assert len(results) >= 1
    assert any("Failed to set pgvector HNSW parameters" in r.message for r in caplog.records)

    # Ensure transaction was recovered and not aborted
    subsequent_res = await integration_db.execute(text("SELECT 1;"))
    assert subsequent_res.scalar() == 1
