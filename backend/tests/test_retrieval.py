import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.main import app
from ai.services.retrieval_service import build_sparse_query

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

