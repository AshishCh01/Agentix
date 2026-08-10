import asyncio
import uuid
import httpx

from app.main import app
from app.auth.dependencies import get_current_user
from app.database.connection import AsyncSessionLocal

USER_A = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "id": "11111111-1111-1111-1111-111111111111",
    "sub": "11111111-1111-1111-1111-111111111111",
    "email": "test_user_a_static@example.com",
    "role": "authenticated"
}

USER_B = {
    "user_id": "22222222-2222-2222-2222-222222222222",
    "id": "22222222-2222-2222-2222-222222222222",
    "sub": "22222222-2222-2222-2222-222222222222",
    "email": "test_user_b_static@example.com",
    "role": "authenticated"
}

def override_get_current_user_a(): return USER_A
def override_get_current_user_b(): return USER_B

async def setup_test_data():
    from app.database.crud import sync_user
    async with AsyncSessionLocal() as db:
        await sync_user(db, USER_A["user_id"], USER_A["email"])
        await sync_user(db, USER_B["user_id"], USER_B["email"])

import pytest

@pytest.fixture(autouse=True)
async def setup_test_users():
    await setup_test_data()
    yield

@pytest.mark.asyncio
async def test_idor_security():
    # Needs ASGI app transport
    transport = httpx.ASGITransport(app=app)
    
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # User A creates a session
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "Test Session User A"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        search_payload = {"session_id": session_id_a, "query": "test query", "top_k": 2}
        search_res = await client.post("/api/v1/search", json=search_payload)
        assert search_res.status_code == 200, f"Got {search_res.status_code}"

        # Switch to User B
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        
        search_res_b = await client.post("/api/v1/search", json=search_payload)
        assert search_res_b.status_code == 404, f"IDOR Vulnerability: {search_res_b.status_code}"

        chat_payload = {"session_id": session_id_a, "message": "Hello"}
        chat_res_b = await client.post("/api/v1/chat", json=chat_payload)
        assert chat_res_b.status_code == 404, f"IDOR Vulnerability: {chat_res_b.status_code}"
