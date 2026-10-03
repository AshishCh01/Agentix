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

        print("Starting test_idor_security")
        search_payload = {"session_id": session_id_a, "query": "test query", "top_k": 2}
        
        from unittest.mock import patch, AsyncMock
        with patch("app.routes.search.search_similar_chunks", new_callable=AsyncMock) as mock_search, \
             patch("app.routes.chat.rag_graph.ainvoke", new_callable=AsyncMock) as mock_chat:
             
            mock_search.return_value = []
            mock_chat.return_value = {"final_response": "Mocked response", "intent": "greeting", "retrieved_chunks": []}
            
            print("Sending POST /api/v1/search (User A)")
            search_res = await client.post("/api/v1/search", json=search_payload)
            print("Received response from /api/v1/search")
            assert search_res.status_code == 200, f"Got {search_res.status_code}"

            # Switch to User B
            app.dependency_overrides[get_current_user] = override_get_current_user_b
            
            print("Sending POST /api/v1/search (User B)")
            search_res_b = await client.post("/api/v1/search", json=search_payload)
            print("Received response from /api/v1/search (User B)")
            assert search_res_b.status_code == 404, f"IDOR Vulnerability: {search_res_b.status_code}"

            print("Sending POST /api/v1/chat (User B)")
            chat_payload = {"session_id": session_id_a, "message": "Hello"}
            chat_res_b = await client.post("/api/v1/chat", json=chat_payload)
            print("Received response from /api/v1/chat (User B)")
            assert chat_res_b.status_code == 404, f"IDOR Vulnerability: {chat_res_b.status_code}"
            print("Done")


@pytest.mark.asyncio
async def test_idor_get_session_by_id():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "User A's Session"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        # Owner can fetch it.
        own_res = await client.get(f"/api/v1/sessions/{session_id_a}")
        assert own_res.status_code == 200

        # A different authenticated user must not be able to.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        other_res = await client.get(f"/api/v1/sessions/{session_id_a}")
        assert other_res.status_code == 404, f"IDOR Vulnerability: {other_res.status_code}"


@pytest.mark.asyncio
async def test_idor_patch_session_title():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "Original Title"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        # User B attempts to rename User A's session.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        patch_res = await client.patch(
            f"/api/v1/sessions/{session_id_a}", json={"title": "Hijacked Title"}
        )
        assert patch_res.status_code == 404, f"IDOR Vulnerability: {patch_res.status_code}"

        # The title must be untouched.
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        check_res = await client.get(f"/api/v1/sessions/{session_id_a}")
        assert check_res.status_code == 200
        assert check_res.json()["title"] == "Original Title"


@pytest.mark.asyncio
async def test_idor_delete_session():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "Do Not Delete Me"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        # User B attempts to delete User A's session.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        delete_res = await client.delete(f"/api/v1/sessions/{session_id_a}")
        assert delete_res.status_code == 404, f"IDOR Vulnerability: {delete_res.status_code}"

        # It must still exist for the owner.
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        check_res = await client.get(f"/api/v1/sessions/{session_id_a}")
        assert check_res.status_code == 200


@pytest.mark.asyncio
async def test_idor_get_session_messages():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "Session With Messages"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        # A different authenticated user must not be able to read the
        # conversation history of a session they don't own.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        messages_res = await client.get(f"/api/v1/sessions/{session_id_a}/messages")
        assert messages_res.status_code == 404, f"IDOR Vulnerability: {messages_res.status_code}"


@pytest.mark.asyncio
async def test_idor_upload_document_to_foreign_session():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/sessions", json={"title": "Target Session"})
        assert res.status_code == 201
        session_id_a = res.json()["id"]

        # User B attempts to attach a document to User A's session by
        # supplying its session_id directly.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        upload_res = await client.post(
            "/api/v1/upload",
            data={"session_id": session_id_a},
            files={"file": ("malicious.txt", b"hello world", "text/plain")},
        )
        assert upload_res.status_code == 404, f"IDOR Vulnerability: {upload_res.status_code}"
