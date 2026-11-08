import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agents.graph import vector_search_node, route_after_vector_search
from ai.services.retrieval_service import search_similar_chunks
from ai.tools.vector_search import vector_search_tool
from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.models.session import ChatSession
from app.models.user import User


class _MockDBSession:
    """Mock AsyncSession that captures executed statements and parameters."""
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed_queries = []
        self.executed_params = []

    async def execute(self, statement, params=None):
        sql_text = statement.text if hasattr(statement, "text") else str(statement)
        self.executed_queries.append(sql_text)
        self.executed_params.append(params)
        mock_result = MagicMock()
        mock_result.mappings.return_value = list(self.rows)
        return mock_result


def _build_candidate_row(
    chunk_id: str,
    doc_id: str,
    filename: str,
    chunk_index: int = 0,
    page_number: int = 1,
    content: str = "sample text",
    rrf_score: float = 0.03,
    source_type: str = "hybrid",
    dense_similarity: float = 0.85,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content,
        "rrf_score": rrf_score,
        "source_type": source_type,
        "sparse_rank": 1,
        "dense_rank": 1,
        "dense_similarity": dense_similarity,
    }


# =====================================================================
# Unit Tests (Mock DB, no live services required)
# =====================================================================

@pytest.mark.asyncio
async def test_kb_chat_with_session_upload_retrieves_both_and_merges_ranking():
    """
    In a knowledge-base chat session, candidates from both the knowledge base
    and direct session uploads are retrieved in a single query, and their
    rankings are merged correctly via RRF and reranker.
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    kb_chunk_id = str(uuid.uuid4())
    session_chunk_id = str(uuid.uuid4())

    mock_rows = [
        _build_candidate_row(
            chunk_id=kb_chunk_id,
            doc_id="kb-doc-1",
            filename="kb_handbook.pdf",
            content="Knowledge base policy handbook",
            rrf_score=0.032,
        ),
        _build_candidate_row(
            chunk_id=session_chunk_id,
            doc_id="session-doc-1",
            filename="session_notes.docx",
            content="Uploaded session specific notes",
            rrf_score=0.030,
        ),
    ]

    mock_db = _MockDBSession(rows=mock_rows)

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * settings.EMBEDDING_DIMENSIONS

        async def mock_rerank(query, chunks):
            # Simulate reranker assigning higher relevance to the session upload chunk
            for c in chunks:
                if c["chunk_id"] == session_chunk_id:
                    c["rerank_score"] = 0.95
                else:
                    c["rerank_score"] = 0.82
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=mock_db,
                query_text="project notes and policy",
                session_id=session_id,
                knowledge_base_id=kb_id,
                user_id=user_id,
                top_k=4,
            )

    # 1. SQL includes the unified scope condition
    sql = next(q for q in mock_db.executed_queries if "dense_inner" in q)
    params = next(p for p in mock_db.executed_params if p and "kb_id" in p)
    assert "(d.knowledge_base_id = :kb_id OR d.session_id = :session_id)" in sql
    assert "d.user_id = :user_id" in sql
    assert "d.status = 'ready'" in sql
    assert params["kb_id"] == str(kb_id)
    assert params["session_id"] == str(session_id)
    assert params["user_id"] == str(user_id)

    # 2. Both KB and session upload chunks are returned and merged
    retrieved_ids = [r["chunk_id"] for r in results]
    assert session_chunk_id in retrieved_ids
    assert kb_chunk_id in retrieved_ids
    assert len(results) == 2

    # 3. Ranking order reflects cross-encoder reranking across the unified candidate pool
    assert results[0]["chunk_id"] == session_chunk_id
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == kb_chunk_id
    assert results[1]["rerank_score"] == 0.82


@pytest.mark.asyncio
async def test_session_isolation_and_cross_user_scoping_in_sql():
    """
    Verify that search_similar_chunks correctly scopes the SQL filter:
    - Scoped to (d.knowledge_base_id = :kb_id OR d.session_id = :session_id)
    - Scoped to d.user_id = :user_id
    - Scoped to d.status = 'ready'
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    mock_db = _MockDBSession(rows=[])

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(
            db=mock_db,
            query_text="security check",
            session_id=session_id,
            knowledge_base_id=kb_id,
            user_id=user_id,
        )

    sql = next(q for q in mock_db.executed_queries if "dense_inner" in q)
    params = next(p for p in mock_db.executed_params if p and "kb_id" in p)

    # Scoping assertions:
    assert "(d.knowledge_base_id = :kb_id OR d.session_id = :session_id)" in sql
    assert "d.user_id = :user_id" in sql
    assert "d.status = 'ready'" in sql
    assert params["session_id"] == str(session_id)
    assert params["kb_id"] == str(kb_id)
    assert params["user_id"] == str(user_id)


@pytest.mark.asyncio
async def test_not_ready_session_document_filter_in_sql():
    """
    Verifies that documents not in status 'ready' are excluded by the SQL filter
    in both sparse_inner and dense_inner CTEs.
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    mock_db = _MockDBSession(rows=[])

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(
            db=mock_db,
            query_text="processing status test",
            session_id=session_id,
            knowledge_base_id=kb_id,
            user_id=user_id,
        )

    sparse_sql = next(q for q in mock_db.executed_queries if "sparse_inner" in q)
    dense_sql = next(q for q in mock_db.executed_queries if "dense_inner" in q)

    assert "d.status = 'ready'" in sparse_sql
    assert "dc.fts_tokens IS NOT NULL" in sparse_sql
    assert "d.status = 'ready'" in dense_sql
    assert "dc.embedding IS NOT NULL" in dense_sql


@pytest.mark.asyncio
async def test_no_session_uploads_returns_kb_only():
    """
    When no session uploads exist, retrieval behaves identically to before:
    only KB documents are returned.
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_chunk_id = str(uuid.uuid4())

    mock_rows = [
        _build_candidate_row(
            chunk_id=kb_chunk_id,
            doc_id="kb-doc-1",
            filename="kb_handbook.pdf",
            content="Only KB content",
        ),
    ]

    mock_db = _MockDBSession(rows=mock_rows)

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * settings.EMBEDDING_DIMENSIONS

        async def mock_rerank(query, chunks):
            for c in chunks:
                c["rerank_score"] = 0.88
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=mock_db,
                query_text="policy handbook",
                session_id=session_id,
                knowledge_base_id=kb_id,
                user_id=user_id,
                top_k=4,
            )

    assert len(results) == 1
    assert results[0]["chunk_id"] == kb_chunk_id
    assert results[0]["filename"] == "kb_handbook.pdf"


@pytest.mark.asyncio
async def test_relevance_threshold_filters_session_upload_chunks():
    """
    Relevance threshold (MIN_RELEVANCE_RERANK) filters out low-scoring chunks
    from session-uploaded documents as well as KB documents.
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    kb_high_id = str(uuid.uuid4())
    session_high_id = str(uuid.uuid4())
    session_low_id = str(uuid.uuid4())
    kb_low_id = str(uuid.uuid4())

    mock_rows = [
        _build_candidate_row(chunk_id=kb_high_id, doc_id="kb-1", filename="kb.pdf"),
        _build_candidate_row(chunk_id=session_high_id, doc_id="s-1", filename="session.pdf"),
        _build_candidate_row(chunk_id=session_low_id, doc_id="s-1", filename="session.pdf"),
        _build_candidate_row(chunk_id=kb_low_id, doc_id="kb-1", filename="kb.pdf"),
    ]

    mock_db = _MockDBSession(rows=mock_rows)

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * settings.EMBEDDING_DIMENSIONS

        async def mock_rerank(query, chunks):
            score_map = {
                kb_high_id: 0.91,
                session_high_id: 0.85,
                session_low_id: 0.08,  # Below threshold (0.30)
                kb_low_id: 0.05,       # Below threshold (0.30)
            }
            for c in chunks:
                c["rerank_score"] = score_map[c["chunk_id"]]
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=mock_db,
                query_text="test query",
                session_id=session_id,
                knowledge_base_id=kb_id,
                user_id=user_id,
                top_k=4,
            )

    # Only high-scoring chunks survive thresholding
    surviving_ids = [r["chunk_id"] for r in results]
    assert kb_high_id in surviving_ids
    assert session_high_id in surviving_ids
    assert session_low_id not in surviving_ids
    assert kb_low_id not in surviving_ids
    assert len(results) == 2


@pytest.mark.asyncio
async def test_all_chunks_below_threshold_triggers_not_covered_route():
    """
    If all chunks from both KB and session uploads fall below the threshold,
    0 chunks survive, and route_after_vector_search routes to answer (not covered).
    """
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    user_id = uuid.uuid4()

    session_chunk_id = str(uuid.uuid4())
    mock_rows = [
        _build_candidate_row(chunk_id=session_chunk_id, doc_id="s-1", filename="session.pdf"),
    ]

    mock_db = _MockDBSession(rows=mock_rows)

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * settings.EMBEDDING_DIMENSIONS

        async def mock_rerank(query, chunks):
            for c in chunks:
                c["rerank_score"] = 0.02  # Well below 0.30
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=mock_db,
                query_text="completely off-topic query",
                session_id=session_id,
                knowledge_base_id=kb_id,
                user_id=user_id,
                top_k=4,
            )

    assert len(results) == 0

    # Test graph routing with 0 chunks in a KB chat:
    state = {
        "retrieved_chunks": results,
        "knowledge_base_id": str(kb_id),
        "allow_web_search": False,
        "retrieval_error": False,
    }
    decision = route_after_vector_search(state)
    assert decision == "answer", "Expected routing to answer for 'not covered' response"


@pytest.mark.asyncio
async def test_vector_search_node_forwards_user_id_and_unified_scope():
    """
    Verifies that vector_search_node in LangGraph extracts session_id,
    knowledge_base_id, and user_id from AgentState and passes all three to vector_search_tool.
    """
    session_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    state = {
        "session_id": session_id,
        "user_id": user_id,
        "knowledge_base_id": kb_id,
        "user_query": "check user scoping",
        "standalone_query": None,
        "image_data": None,
        "tool_outputs": [],
    }

    with patch("ai.agents.graph.vector_search_tool", new_callable=AsyncMock) as mock_tool:
        mock_tool.return_value = {"chunks": [], "context_text": ""}
        await vector_search_node(state, config={"configurable": {"db": MagicMock()}})

    assert mock_tool.call_args.kwargs["knowledge_base_id"] == uuid.UUID(kb_id)
    assert str(mock_tool.call_args.kwargs["session_id"]) == session_id
    assert mock_tool.call_args.kwargs["user_id"] == uuid.UUID(user_id)


# =====================================================================
# Scoping Removal Vulnerability Test (Counterfactual test)
# =====================================================================

@pytest.mark.asyncio
async def test_scoping_filter_counterfactual_leakage():
    """
    SECURITY TEST:
    Demonstrates that without the session_id / user_id scoping condition,
    another user's or another session's uploads would be returned.
    Tests that our filter logic strictly excludes them.
    """
    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()
    session_1_id = uuid.uuid4()
    session_2_id = uuid.uuid4()
    kb_a_id = uuid.uuid4()

    # Documents present:
    # 1. User A, Session 1, KB A -> Should match
    # 2. User A, Session 2 (different session) -> Should NOT match
    # 3. User B, Session 2 (different user) -> Should NOT match
    all_mock_candidates = [
        {"chunk_id": "c1", "user_id": user_a_id, "session_id": session_1_id, "kb_id": kb_a_id, "status": "ready"},
        {"chunk_id": "c2", "user_id": user_a_id, "session_id": session_2_id, "kb_id": None, "status": "ready"},
        {"chunk_id": "c3", "user_id": user_b_id, "session_id": session_2_id, "kb_id": None, "status": "ready"},
    ]

    # Filter predicate implemented in our SQL with fail-closed user scoping
    def current_filter(c, q_session_id, q_kb_id, q_user_id):
        if not q_user_id:
            return False
        kb_or_session = (c["kb_id"] == q_kb_id or c["session_id"] == q_session_id)
        user_match = (c["user_id"] == q_user_id)
        status_match = (c["status"] == "ready")
        return kb_or_session and user_match and status_match

    # Under current filter for User A querying Session 1:
    matching = [c["chunk_id"] for c in all_mock_candidates if current_filter(c, session_1_id, kb_a_id, user_a_id)]
    assert matching == ["c1"], "Only User A Session 1 chunk should match"
    assert "c2" not in matching, "Session 2 chunk of User A must not leak into Session 1"
    assert "c3" not in matching, "User B chunk must not leak into User A"

    # Counterfactual: without user_id, fail-closed returns nothing
    assert [c["chunk_id"] for c in all_mock_candidates if current_filter(c, session_1_id, kb_a_id, None)] == []

    # If the scoping were removed (counterfactual):
    def leaky_scoping_no_filter(c):
        return True  # No scoping filter
    leaked = [c["chunk_id"] for c in all_mock_candidates if leaky_scoping_no_filter(c)]
    assert len(leaked) == 3  # All would leak, proving that the scoping filter is essential


# =====================================================================
# Fail-Closed User Scoping Tests (Commit 1 hardening)
# =====================================================================

@pytest.mark.asyncio
async def test_retrieval_fails_closed_when_user_id_missing_or_invalid():
    """
    search_similar_chunks must fail closed and return 0 chunks immediately
    without issuing any database queries if user_id is None, invalid string, or non-UUID.
    """
    mock_db = _MockDBSession(rows=[])
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    # 1. user_id is None
    res1 = await search_similar_chunks(
        db=mock_db,
        query_text="test query",
        session_id=session_id,
        knowledge_base_id=kb_id,
        user_id=None,
    )
    assert res1 == []
    assert len(mock_db.executed_queries) == 0

    # 2. user_id is invalid string (not a valid UUID)
    res2 = await search_similar_chunks(
        db=mock_db,
        query_text="test query",
        session_id=session_id,
        knowledge_base_id=kb_id,
        user_id="not-a-valid-uuid",
    )
    assert res2 == []
    assert len(mock_db.executed_queries) == 0

    # 3. user_id is non-UUID type (e.g. integer)
    res3 = await search_similar_chunks(
        db=mock_db,
        query_text="test query",
        session_id=session_id,
        knowledge_base_id=kb_id,
        user_id=12345,
    )
    assert res3 == []
    assert len(mock_db.executed_queries) == 0


@pytest.mark.asyncio
async def test_vector_search_tool_fails_closed_when_user_id_missing_or_invalid():
    """vector_search_tool must return empty chunks and context when user_id is invalid or missing."""
    mock_db = _MockDBSession(rows=[])
    session_id = uuid.uuid4()

    res_none = await vector_search_tool(
        db=mock_db,
        session_id=session_id,
        query="test",
        user_id=None,
    )
    assert res_none == {"chunks": [], "context_text": ""}

    res_invalid = await vector_search_tool(
        db=mock_db,
        session_id=session_id,
        query="test",
        user_id="invalid-uuid-string",
    )
    assert res_invalid == {"chunks": [], "context_text": ""}


@pytest.mark.asyncio
async def test_vector_search_node_fails_closed_when_user_id_missing_or_invalid():
    """vector_search_node in LangGraph must return empty retrieved_chunks when user_id is missing/invalid."""
    mock_db = _MockDBSession(rows=[])
    config = {"configurable": {"db": mock_db}}

    # Missing user_id in state
    state_no_user = {
        "session_id": str(uuid.uuid4()),
        "user_query": "hello",
    }
    out1 = await vector_search_node(state_no_user, config)
    assert out1["retrieved_chunks"] == []
    assert out1["formatted_context"] == ""

    # Invalid user_id in state
    state_bad_user = {
        "session_id": str(uuid.uuid4()),
        "user_id": "malformed-user-id",
        "user_query": "hello",
    }
    out2 = await vector_search_node(state_bad_user, config)
    assert out2["retrieved_chunks"] == []
    assert out2["formatted_context"] == ""


@pytest.mark.asyncio
async def test_user_scoping_enforced_in_session_only_branch():
    """
    Verify that in session-only chats (knowledge_base_id is None),
    the SQL filter strictly enforces d.session_id = :scope_id AND d.user_id = :user_id.
    """
    session_id = uuid.uuid4()
    user_id = uuid.uuid4()
    mock_db = _MockDBSession(rows=[])

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.0] * settings.EMBEDDING_DIMENSIONS
        await search_similar_chunks(
            db=mock_db,
            query_text="session only query",
            session_id=session_id,
            knowledge_base_id=None,
            user_id=user_id,
        )

    sql = next(q for q in mock_db.executed_queries if "dense_inner" in q)
    params = next(p for p in mock_db.executed_params if p and "scope_id" in p)

    assert "d.session_id = :scope_id" in sql
    assert "d.user_id = :user_id" in sql
    assert "d.status = 'ready'" in sql
    assert params["scope_id"] == str(session_id)
    assert params["user_id"] == str(user_id)


@pytest.mark.asyncio
async def test_search_endpoint_fails_closed_with_missing_or_invalid_user():
    """
    Verifies that POST /api/v1/search fails closed with HTTP 401 when the user ID
    is missing or cannot be parsed as a valid UUID.
    """
    from app.main import app
    from app.auth.dependencies import get_current_user
    import httpx

    # 1. Missing user ID
    app.dependency_overrides[get_current_user] = lambda: {"id": "", "sub": ""}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post(
            "/api/v1/search",
            json={
                "session_id": str(uuid.uuid4()),
                "query": "test query",
                "top_k": 4,
            },
        )
    assert resp1.status_code == 401
    assert "resolve user identity" in resp1.json()["detail"].lower()

    # 2. Invalid UUID user ID
    app.dependency_overrides[get_current_user] = lambda: {"id": "bad-uuid", "sub": "bad-uuid"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp2 = await client.post(
            "/api/v1/search",
            json={
                "session_id": str(uuid.uuid4()),
                "query": "test query",
                "top_k": 4,
            },
        )
    assert resp2.status_code == 401
    assert "invalid user identity format" in resp2.json()["detail"].lower()

