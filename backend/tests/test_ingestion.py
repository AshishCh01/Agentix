import pytest
import httpx
from unittest.mock import patch, MagicMock
from app.main import app
from app.auth.dependencies import get_current_user
from app.database.connection import AsyncSessionLocal
from app.database.crud import sync_user

USER_A = {
    "user_id": "33333333-3333-3333-3333-333333333333",
    "id": "33333333-3333-3333-3333-333333333333",
    "sub": "33333333-3333-3333-3333-333333333333",
    "email": "ingestion_test_static@example.com",
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
async def test_ingestion_endpoints():
    transport = httpx.ASGITransport(app=app)
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = override_get_current_user_a
            
            # 1. Create a session
            res = await client.post("/api/v1/sessions", json={"title": "Test Session"})
            assert res.status_code == 201
            session_id = res.json()["id"]

            # 2. Upload a normal text file
            normal_file_content = b"This is a normal text file. It has enough content to be chunked. " * 50
            files = {"file": ("normal.txt", normal_file_content, "text/plain")}
            data = {"session_id": session_id}
            
            res = await client.post("/api/v1/upload", data=data, files=files)
            assert res.status_code == 202

            # 3. Upload an empty file
            empty_files = {"file": ("empty.txt", b"   \n  \t ", "text/plain")}
            res = await client.post("/api/v1/upload", data=data, files=empty_files)
            assert res.status_code == 202

            # 4. Upload an unsupported binary file
            bad_files = {"file": ("bad.bin", b"\x00\x01\x02\x03\x04\x05\xFF\xFE", "application/octet-stream")}
            res = await client.post("/api/v1/upload", data=data, files=bad_files)
            assert res.status_code == 202

            # 5. Upload a pseudo-supported binary without extension
            pseudo_files = {"file": ("no_extension", b"I am actually text but I have no extension", "application/octet-stream")}
            res = await client.post("/api/v1/upload", data=data, files=pseudo_files)
            assert res.status_code == 202
            
            # Verify background task was called 4 times
            assert mock_bg.call_count == 4