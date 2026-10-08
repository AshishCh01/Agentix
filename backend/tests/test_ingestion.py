import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
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

from app.database.connection import get_db

async def override_get_db():
    mock_db = AsyncMock()
    
    # Mock execute to return a fake result for scalar_one_or_none
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = MagicMock(id="mocked-id")
    mock_db.execute.return_value = mock_result
    
    async def mock_refresh(instance):
        from datetime import datetime, timezone
        import uuid
        if not getattr(instance, "id", None):
            instance.id = uuid.uuid4()
        instance.created_at = datetime.now(timezone.utc)
        instance.updated_at = datetime.now(timezone.utc)
    
    mock_db.refresh = AsyncMock(side_effect=mock_refresh)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    
    yield mock_db

@pytest.mark.asyncio
async def test_ingestion_endpoints():
    transport = httpx.ASGITransport(app=app)
    
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg, \
         patch("app.routes.upload.get_supabase_client"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = override_get_current_user_a
            app.dependency_overrides[get_db] = override_get_db
            
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