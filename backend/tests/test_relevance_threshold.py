import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.agents.answer import run_answer_agent
from ai.agents.graph import route_after_vector_search, route_intent
from ai.services.retrieval_service import search_similar_chunks
from app.config.settings import Settings, settings


@pytest.mark.asyncio
async def test_threshold_filter_above_kept_and_below_dropped():
    """Chunks with rerank_score >= MIN_RELEVANCE_RERANK are kept; those below are dropped."""
    db_mock = AsyncMock()

    mock_rows = [
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Relevant content",
            "rrf_score": 0.03,
            "source_type": "hybrid",
            "sparse_rank": 1,
            "dense_rank": 1,
            "dense_similarity": 0.85,
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Irrelevant content",
            "rrf_score": 0.02,
            "source_type": "hybrid",
            "sparse_rank": 2,
            "dense_rank": 2,
            "dense_similarity": 0.50,
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 3,
            "chunk_index": 2,
            "content": "Highly relevant content",
            "rrf_score": 0.04,
            "source_type": "hybrid",
            "sparse_rank": 1,
            "dense_rank": 1,
            "dense_similarity": 0.90,
        },
    ]

    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_rows
    db_mock.execute.return_value = mock_result

    with patch.object(settings, "RERANK_PROVIDER", "cross-encoder"), \
         patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 2000

        async def mock_rerank(query, chunks):
            scores = [0.85, 0.005, 0.95]
            for i, c in enumerate(chunks):
                c["rerank_score"] = scores[i]
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=db_mock,
                query_text="test query",
                session_id=uuid.uuid4(),
                top_k=4,
            )

            # Kept: 0.95 and 0.85 (>= 0.30). Dropped: 0.005 (< 0.30).
            assert len(results) == 2
            assert results[0]["rerank_score"] == 0.95
            assert results[1]["rerank_score"] == 0.85


@pytest.mark.asyncio
async def test_threshold_filter_boundary_value():
    """Boundary test: chunk with rerank_score exactly equal to MIN_RELEVANCE_RERANK is kept, just below is dropped."""
    db_mock = AsyncMock()
    mock_rows = [
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 1,
            "chunk_index": 0,
            "content": "At boundary",
            "rrf_score": 0.03,
            "source_type": "hybrid",
            "sparse_rank": 1,
            "dense_rank": 1,
            "dense_similarity": 0.70,
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Just below boundary",
            "rrf_score": 0.02,
            "source_type": "hybrid",
            "sparse_rank": 2,
            "dense_rank": 2,
            "dense_similarity": 0.69,
        },
    ]
    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_rows
    db_mock.execute.return_value = mock_result

    with patch.object(settings, "RERANK_PROVIDER", "cross-encoder"), \
         patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 2000

        async def mock_rerank(query, chunks):
            chunks[0]["rerank_score"] = 0.3000
            chunks[1]["rerank_score"] = 0.2999
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=db_mock,
                query_text="boundary query",
                session_id=uuid.uuid4(),
                top_k=4,
            )

            assert len(results) == 1
            assert results[0]["rerank_score"] == 0.3000


@pytest.mark.asyncio
async def test_threshold_filter_all_chunks_dropped():
    """When all chunks score below threshold, an empty list is returned."""
    db_mock = AsyncMock()
    mock_rows = [
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Off-topic chunk 1",
            "rrf_score": 0.01,
            "source_type": "dense",
            "sparse_rank": None,
            "dense_rank": 10,
            "dense_similarity": 0.40,
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Off-topic chunk 2",
            "rrf_score": 0.01,
            "source_type": "dense",
            "sparse_rank": None,
            "dense_rank": 11,
            "dense_similarity": 0.35,
        },
    ]
    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_rows
    db_mock.execute.return_value = mock_result

    with patch.object(settings, "RERANK_PROVIDER", "cross-encoder"), \
         patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 2000

        async def mock_rerank(query, chunks):
            chunks[0]["rerank_score"] = 0.0050
            chunks[1]["rerank_score"] = 0.0001
            return chunks

        with patch("ai.services.retrieval_service.embedding_service.rerank_chunks", side_effect=mock_rerank):
            results = await search_similar_chunks(
                db=db_mock,
                query_text="completely off-topic query",
                session_id=uuid.uuid4(),
                top_k=4,
            )

            assert results == []


@pytest.mark.asyncio
async def test_threshold_filter_fallback_when_rerank_disabled():
    """When RERANK_PROVIDER is 'none', fallback to MIN_RELEVANCE_DENSE on dense_similarity."""
    db_mock = AsyncMock()
    mock_rows = [
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Dense relevant chunk",
            "rrf_score": 0.03,
            "source_type": "dense",
            "sparse_rank": None,
            "dense_rank": 1,
            "dense_similarity": 0.75,  # > 0.68
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Dense at boundary chunk",
            "rrf_score": 0.02,
            "source_type": "dense",
            "sparse_rank": None,
            "dense_rank": 2,
            "dense_similarity": 0.68,  # == 0.68
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "filename": "doc1.docx",
            "page_number": 3,
            "chunk_index": 2,
            "content": "Dense irrelevant chunk",
            "rrf_score": 0.01,
            "source_type": "dense",
            "sparse_rank": None,
            "dense_rank": 3,
            "dense_similarity": 0.61,  # < 0.68 (unanswerable range in Phase 1)
        },
    ]
    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_rows
    db_mock.execute.return_value = mock_result

    with patch("ai.services.retrieval_service.embedding_service.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 2000

        with patch.object(settings, "RERANK_PROVIDER", "none"):
            # When rerank is none, rerank_chunks returns chunks untouched (no rerank_score)
            results = await search_similar_chunks(
                db=db_mock,
                query_text="dense query",
                session_id=uuid.uuid4(),
                top_k=4,
            )

            # Kept: 0.75 and 0.68; Dropped: 0.61
            assert len(results) == 2
            assert results[0]["dense_similarity"] == 0.75
            assert results[1]["dense_similarity"] == 0.68


def test_env_override_settings(monkeypatch):
    """Verifies that MIN_RELEVANCE_RERANK and MIN_RELEVANCE_DENSE are overridable via environment variables."""
    monkeypatch.setenv("MIN_RELEVANCE_RERANK", "0.45")
    monkeypatch.setenv("MIN_RELEVANCE_DENSE", "0.72")
    custom_settings = Settings()
    assert custom_settings.MIN_RELEVANCE_RERANK == 0.45
    assert custom_settings.MIN_RELEVANCE_DENSE == 0.72


def test_route_after_vector_search_kb_empty_routes_to_answer():
    """When 0 chunks pass threshold in a KB chat without opt-in, route to 'answer'."""
    state = {
        "knowledge_base_id": str(uuid.uuid4()),
        "retrieved_chunks": [],
        "allow_web_search": False,
        "retrieval_error": False,
    }
    assert route_after_vector_search(state) == "answer"


def test_route_after_vector_search_kb_opt_in_routes_to_web_search():
    """When 0 chunks pass threshold in a KB chat WITH opt-in, route to 'web_search'."""
    state = {
        "knowledge_base_id": str(uuid.uuid4()),
        "retrieved_chunks": [],
        "allow_web_search": True,
        "retrieval_error": False,
    }
    assert route_after_vector_search(state) == "web_search"


def test_route_after_vector_search_kb_with_chunks_routes_to_answer():
    """When chunks pass threshold in a KB chat, route to 'answer' regardless of opt-in flag."""
    state = {
        "knowledge_base_id": str(uuid.uuid4()),
        "retrieved_chunks": [{"chunk_id": "c1", "content": "valid"}],
        "allow_web_search": True,
        "retrieval_error": False,
    }
    assert route_after_vector_search(state) == "answer"


def test_route_after_vector_search_non_kb_preserves_web_search_fallback():
    """In a chat with no knowledge base attached, empty chunks still routes to 'web_search'."""
    state = {
        "knowledge_base_id": None,
        "retrieved_chunks": [],
        "retrieval_error": False,
    }
    assert route_after_vector_search(state) == "web_search"


def test_supervisor_web_search_intent_routing_preserved():
    """Current-information queries classified as WEB_SEARCH by supervisor route directly to web_search."""
    state = {
        "intent": "WEB_SEARCH",
    }
    assert route_intent(state) == "web_search"


@pytest.mark.asyncio
async def test_answer_agent_not_covered_response_without_llm():
    """
    When no context is retrieved for document search, answer agent returns the
    not covered message offering web search, without calling the LLM.
    """
    state = {
        "formatted_context": "",
        "image_data": None,
        "context_source": "document",
        "retrieval_error": False,
    }
    with patch("ai.agents.answer.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
        res = await run_answer_agent(state)
        mock_llm.assert_not_called()
        assert "I couldn't find this in your uploaded documents." in res["final_response"]
        assert "Would you like me to search the web instead?" in res["final_response"]

