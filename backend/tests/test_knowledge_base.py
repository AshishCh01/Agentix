import uuid

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.main import app
from app.auth.dependencies import get_current_user
from app.database import crud
from app.database.connection import AsyncSessionLocal

USER_A = {
    "user_id": "44444444-4444-4444-4444-444444444444",
    "id": "44444444-4444-4444-4444-444444444444",
    "sub": "44444444-4444-4444-4444-444444444444",
    "email": "kb_test_user_a@example.com",
    "role": "authenticated",
}

USER_B = {
    "user_id": "55555555-5555-5555-5555-555555555555",
    "id": "55555555-5555-5555-5555-555555555555",
    "sub": "55555555-5555-5555-5555-555555555555",
    "email": "kb_test_user_b@example.com",
    "role": "authenticated",
}


def override_get_current_user_a():
    return USER_A


def override_get_current_user_b():
    return USER_B


@pytest.fixture(autouse=True)
async def setup_test_users():
    async with AsyncSessionLocal() as db:
        await crud.sync_user(db, USER_A["user_id"], USER_A["email"])
        await crud.sync_user(db, USER_B["user_id"], USER_B["email"])
    yield


@pytest.mark.asyncio
async def test_knowledge_base_crud_lifecycle():
    """Create -> list -> get -> patch -> delete, all via the HTTP API."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a

        # Create
        res = await client.post(
            "/api/v1/knowledge-bases",
            json={"name": "Product Docs", "description": "Internal product documentation"},
        )
        assert res.status_code == 201
        kb = res.json()
        assert kb["name"] == "Product Docs"
        assert kb["description"] == "Internal product documentation"
        assert kb["document_count"] == 0
        assert kb["user_id"] == USER_A["user_id"]
        kb_id = kb["id"]

        # List
        res = await client.get("/api/v1/knowledge-bases")
        assert res.status_code == 200
        assert any(item["id"] == kb_id for item in res.json())

        # Get
        res = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 200
        assert res.json()["name"] == "Product Docs"

        # Patch (rename)
        res = await client.patch(
            f"/api/v1/knowledge-bases/{kb_id}", json={"name": "Renamed Docs"}
        )
        assert res.status_code == 200
        assert res.json()["name"] == "Renamed Docs"
        # Description untouched by a partial update
        assert res.json()["description"] == "Internal product documentation"

        # Delete
        res = await client.delete(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 204

        # Now gone
        res = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_knowledge_base_create_requires_nonempty_name():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/knowledge-bases", json={"name": ""})
        assert res.status_code == 422


@pytest.mark.asyncio
async def test_knowledge_base_ownership_is_enforced():
    """
    IDOR protection: a knowledge base created by USER_A must be invisible
    (404, not 403 -- existence itself is not disclosed) to USER_B across
    every endpoint: get, patch, delete, list-documents, upload-document.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post(
            "/api/v1/knowledge-bases", json={"name": "USER_A Private KB"}
        )
        assert res.status_code == 201
        kb_id = res.json()["id"]

        # Switch identity to USER_B
        app.dependency_overrides[get_current_user] = override_get_current_user_b

        res = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 404

        res = await client.patch(
            f"/api/v1/knowledge-bases/{kb_id}", json={"name": "Hijacked"}
        )
        assert res.status_code == 404

        res = await client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
        assert res.status_code == 404

        with patch("app.routes.upload.get_supabase_client"):
            files = {"file": ("doc.txt", b"some content", "text/plain")}
            res = await client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files=files)
        assert res.status_code == 404

        res = await client.delete(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 404

        # USER_A's KB must be untouched by all of the above.
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
        assert res.status_code == 200
        assert res.json()["name"] == "USER_A Private KB"

        # cleanup
        await client.delete(f"/api/v1/knowledge-bases/{kb_id}")


@pytest.mark.asyncio
async def test_knowledge_base_list_and_get_do_not_leak_other_users_kbs():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        res = await client.post("/api/v1/knowledge-bases", json={"name": "USER_B KB"})
        assert res.status_code == 201
        b_kb_id = res.json()["id"]

        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.get("/api/v1/knowledge-bases")
        assert res.status_code == 200
        assert all(item["id"] != b_kb_id for item in res.json())

        app.dependency_overrides[get_current_user] = override_get_current_user_b
        await client.delete(f"/api/v1/knowledge-bases/{b_kb_id}")


@pytest.mark.asyncio
async def test_document_upload_associates_with_knowledge_base():
    """
    Uploading a document into a knowledge base must create a Document row
    tagged with that knowledge_base_id (and no session_id), and it must show
    up in the KB's document list with a 'processing' status.
    """
    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg, \
         patch("app.routes.upload.get_supabase_client"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = override_get_current_user_a

            res = await client.post(
                "/api/v1/knowledge-bases", json={"name": "Ingestion Target KB"}
            )
            assert res.status_code == 201
            kb_id = res.json()["id"]

            file_content = b"Knowledge base scoped document content. " * 20
            files = {"file": ("kb_doc.txt", file_content, "text/plain")}
            res = await client.post(f"/api/v1/knowledge-bases/{kb_id}/documents", files=files)
            assert res.status_code == 202
            body = res.json()
            assert body["status"] == "processing"
            document_id = body["document_id"]

            # Background ingestion was scheduled, not run synchronously.
            assert mock_bg.call_count == 1

            res = await client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
            assert res.status_code == 200
            docs = res.json()["documents"]
            assert len(docs) == 1
            assert docs[0]["id"] == document_id
            assert docs[0]["knowledge_base_id"] == kb_id
            assert docs[0]["session_id"] is None
            assert docs[0]["status"] == "processing"

            # KB's document_count reflects the association.
            res = await client.get(f"/api/v1/knowledge-bases/{kb_id}")
            assert res.json()["document_count"] == 1

            # Delete the document explicitly.
            res = await client.delete(f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}")
            assert res.status_code == 204

            res = await client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
            assert res.json()["documents"] == []

            await client.delete(f"/api/v1/knowledge-bases/{kb_id}")


@pytest.mark.asyncio
async def test_delete_knowledge_base_cascades_to_documents():
    """Deleting a KB must delete its documents at the database level (ON DELETE CASCADE)."""
    async with AsyncSessionLocal() as db:
        kb = await crud.create_knowledge_base(db, user_id=USER_A["user_id"], name="Cascade Test KB")

        from app.models.document import Document

        doc = Document(
            user_id=uuid.UUID(USER_A["user_id"]),
            session_id=None,
            knowledge_base_id=kb.id,
            filename="cascade.txt",
            file_type="text/plain",
            file_path="fake/path.txt",
            file_size=10,
            status="ready",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        doc_id = doc.id

        deleted = await crud.delete_knowledge_base(db, knowledge_base_id=kb.id, user_id=USER_A["user_id"])
        assert deleted is True

        from sqlalchemy import select

        result = await db.execute(select(Document).where(Document.id == doc_id))
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_chat_session_can_be_scoped_to_a_knowledge_base():
    """
    Creating a chat session with knowledge_base_id persists the association,
    and a non-owned knowledge_base_id is rejected with 404 (IDOR
    protection on session creation, not just on the KB endpoints).
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post("/api/v1/knowledge-bases", json={"name": "Chat-Scoped KB"})
        kb_id = res.json()["id"]

        res = await client.post(
            "/api/v1/sessions", json={"title": "Scoped Chat", "knowledge_base_id": kb_id}
        )
        assert res.status_code == 201
        assert res.json()["knowledge_base_id"] == kb_id

        # A KB owned by another user must be rejected.
        app.dependency_overrides[get_current_user] = override_get_current_user_b
        res = await client.post(
            "/api/v1/knowledge-bases", json={"name": "USER_B KB For Chat Test"}
        )
        b_kb_id = res.json()["id"]

        app.dependency_overrides[get_current_user] = override_get_current_user_a
        res = await client.post(
            "/api/v1/sessions",
            json={"title": "Should Fail", "knowledge_base_id": b_kb_id},
        )
        assert res.status_code == 404

        app.dependency_overrides[get_current_user] = override_get_current_user_b
        await client.delete(f"/api/v1/knowledge-bases/{b_kb_id}")
        app.dependency_overrides[get_current_user] = override_get_current_user_a
        await client.delete(f"/api/v1/knowledge-bases/{kb_id}")


@pytest.mark.asyncio
async def test_vector_search_tool_forwards_knowledge_base_scope():
    """
    vector_search_node must read knowledge_base_id off graph state and pass
    it through to vector_search_tool -- this is what makes a KB-scoped chat
    retrieve only that KB's documents instead of falling back to
    session-scoped retrieval.
    """
    from ai.agents.graph import vector_search_node

    kb_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    state = {
        "session_id": session_id,
        "user_id": USER_A["user_id"],
        "knowledge_base_id": kb_id,
        "user_query": "what does the doc say?",
        "standalone_query": None,
        "image_data": None,
        "tool_outputs": [],
    }

    with patch("ai.agents.graph.vector_search_tool", new_callable=AsyncMock) as mock_tool:
        mock_tool.return_value = {"chunks": [], "context_text": ""}
        await vector_search_node(state, config={"configurable": {"db": object()}})

    assert mock_tool.call_args.kwargs["knowledge_base_id"] == uuid.UUID(kb_id)
    assert str(mock_tool.call_args.kwargs["session_id"]) == session_id


@pytest.mark.asyncio
async def test_search_similar_chunks_scopes_sql_by_knowledge_base():
    """
    search_similar_chunks must filter on d.knowledge_base_id (not
    d.session_id) when a knowledge_base_id is supplied, and must fall back
    to the existing d.session_id filter when it is not -- preserving prior
    session-scoped retrieval behavior for chats with no KB attached.
    """
    from ai.services.retrieval_service import search_similar_chunks

    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    class _FakeResult:
        def mappings(self):
            return []

    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_FakeResult())

    with patch(
        "ai.services.retrieval_service.embedding_service.generate_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [0.0] * 8

        # KB-scoped call
        await search_similar_chunks(
            db=fake_db, query_text="q", session_id=session_id, knowledge_base_id=kb_id
        )
        sql_text, params = fake_db.execute.call_args.args
        assert "d.knowledge_base_id = :scope_id" in str(sql_text)
        assert "d.session_id = :scope_id" not in str(sql_text)
        assert params["scope_id"] == str(kb_id)

        # Session-scoped call (default/backward-compatible behavior)
        fake_db.execute.reset_mock()
        await search_similar_chunks(db=fake_db, query_text="q", session_id=session_id)
        sql_text, params = fake_db.execute.call_args.args
        assert "d.session_id = :scope_id" in str(sql_text)
        assert "d.knowledge_base_id = :scope_id" not in str(sql_text)
        assert params["scope_id"] == str(session_id)
