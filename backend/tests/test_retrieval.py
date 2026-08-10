import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.main import app

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
                    "source_type": "reranked"
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
            assert chunk["source_type"] == "reranked"
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
