import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.main import app
from app.auth.dependencies import get_current_user

from app.database.connection import AsyncSessionLocal
from app.database.crud import sync_user

USER_A = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "id": "11111111-1111-1111-1111-111111111111",
    "sub": "11111111-1111-1111-1111-111111111111",
    "email": "test_user_a_static@example.com",
    "role": "authenticated"
}

def override_get_current_user_a(): return USER_A

async def setup_test_data():
    async with AsyncSessionLocal() as db:
        await sync_user(db, USER_A["user_id"], USER_A["email"])

@pytest.fixture(autouse=True)
async def setup_test_users():
    await setup_test_data()
    yield


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
        
        async def dummy_events(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "supervisor"}
            yield {"event": "on_chat_model_stream", "data": {"chunk": type("Chunk", (), {"content": "Mocked"})()}}
            yield {"event": "on_chain_end", "name": "rag_graph", "data": {"output": {"final_response": "Mocked", "intent": "greeting", "retrieved_chunks": []}}}
            
        with patch("app.routes.chat.rag_graph.astream_events", side_effect=dummy_events):
            async with client.stream("POST", "/api/v1/chat/stream", json=chat_payload) as response:
                assert response.status_code == 200
                
                # Read the entire SSE stream
                stream_output = await response.aread()
                assert b"data:" in stream_output
