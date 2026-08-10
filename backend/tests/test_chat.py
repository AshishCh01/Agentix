import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.main import app
from app.auth.dependencies import get_current_user

USER_A = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "id": "11111111-1111-1111-1111-111111111111",
    "sub": "11111111-1111-1111-1111-111111111111",
    "email": "test_user_a_static@example.com",
    "role": "authenticated"
}

def override_get_current_user_a(): return USER_A

@pytest.mark.asyncio
async def test_chat_stream_endpoint():
    transport = httpx.ASGITransport(app=app)
    
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        
        # 1. Create a session
        res = await client.post("/api/v1/sessions", json={"title": "Test Chat Session"})
        assert res.status_code == 201
        session_id = res.json()["id"]

        # 2. Test chat stream endpoint
        chat_payload = {"session_id": session_id, "message": "Hello, this is a test!"}
        
        with patch("ai.agents.greeting.llm_service.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Hello! I am a mocked greeting response."
            
            async with client.stream("POST", "/api/v1/chat/stream", json=chat_payload) as response:
                assert response.status_code == 200
                
                # Read the entire SSE stream
                stream_output = await response.aread()
                assert b"data:" in stream_output
