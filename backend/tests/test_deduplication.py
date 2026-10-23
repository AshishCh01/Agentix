import uuid
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from sqlalchemy.exc import IntegrityError

from app.main import app
from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.models.document import Document
from app.models.session import ChatSession
from app.models.knowledge_base import KnowledgeBase
from app.models.document_chunks import DocumentChunk
from app.routes.upload import process_document_background

USER_A = {
    "user_id": "11111111-1111-1111-1111-111111111111",
    "id": "11111111-1111-1111-1111-111111111111",
    "sub": "11111111-1111-1111-1111-111111111111",
    "email": "user_a@example.com",
    "role": "authenticated",
}

USER_B = {
    "user_id": "22222222-2222-2222-2222-222222222222",
    "id": "22222222-2222-2222-2222-222222222222",
    "sub": "22222222-2222-2222-2222-222222222222",
    "email": "user_b@example.com",
    "role": "authenticated",
}


class MockDBStore:
    """In-memory mock store that simulates PostgreSQL uniqueness and query semantics."""

    def __init__(self):
        self.sessions = {}
        self.kbs = {}
        self.documents = {}
        self.chunks = []
        self.fail_commit_with_integrity_error = False
        self.race_condition_mode = False
        self._doc_query_count = 0

    def create_session(self, user_id, session_id=None, kb_id=None):
        sid = session_id or uuid.uuid4()
        session = ChatSession(id=sid, user_id=user_id, title="Test Session", knowledge_base_id=kb_id)
        self.sessions[sid] = session
        return session

    def create_kb(self, user_id, kb_id=None):
        k_id = kb_id or uuid.uuid4()
        kb = KnowledgeBase(id=k_id, user_id=user_id, name="Test KB")
        self.kbs[k_id] = kb
        return kb

    def get_session(self, session_id, user_id):
        sess = self.sessions.get(session_id)
        if sess and sess.user_id == user_id:
            return sess
        return None

    def get_kb(self, kb_id, user_id):
        kb = self.kbs.get(kb_id)
        if kb and kb.user_id == user_id:
            return kb
        return None

    def find_duplicate_document(self, user_id, content_hash, session_id=None, knowledge_base_id=None):
        if self.race_condition_mode and self._doc_query_count == 0:
            # First query is the pre-check: simulate that another transaction hasn't committed yet
            self._doc_query_count += 1
            return None

        for doc in self.documents.values():
            if (
                doc.user_id == user_id
                and doc.content_hash == content_hash
                and doc.session_id == session_id
                and doc.knowledge_base_id == knowledge_base_id
            ):
                return doc
        return None


class MockAsyncSession:
    def __init__(self, store: MockDBStore):
        self.store = store
        self.pending = []

    async def execute(self, stmt):
        stmt_str = str(stmt).lower()
        mock_res = MagicMock()

        # Query on ChatSession
        if "chat_sessions" in stmt_str:
            params = stmt.compile().params
            user_id = params.get("user_id_1")
            session_id = params.get("id_1") or params.get("session_id_1")
            sess = self.store.get_session(session_id, user_id)
            mock_res.scalar_one_or_none.return_value = sess
            return mock_res

        # Query on KnowledgeBase
        if "knowledge_bases" in stmt_str:
            params = stmt.compile().params
            user_id = params.get("user_id_1")
            kb_id = params.get("id_1") or params.get("knowledge_base_id_1")
            kb = self.store.get_kb(kb_id, user_id)
            mock_res.scalar_one_or_none.return_value = kb
            return mock_res

        # Query on Document
        if "documents" in stmt_str:
            params = stmt.compile().params
            user_id = params.get("user_id_1")
            content_hash = params.get("content_hash_1")
            session_id = params.get("session_id_1")
            kb_id = params.get("knowledge_base_id_1")

            # Check if this is a find-by-id query
            if "documents.id" in stmt_str and "documents.content_hash" not in stmt_str:
                doc_id = params.get("id_1") or params.get("document_id_1")
                doc = self.store.documents.get(doc_id)
                mock_res.scalar_one_or_none.return_value = doc
                return mock_res

            doc = self.store.find_duplicate_document(
                user_id=user_id,
                content_hash=content_hash,
                session_id=session_id,
                knowledge_base_id=kb_id,
            )
            mock_res.scalar_one_or_none.return_value = doc
            return mock_res

        mock_res.scalar_one_or_none.return_value = None
        return mock_res

    def add(self, instance):
        self.pending.append(instance)

    async def commit(self):
        if self.store.fail_commit_with_integrity_error:
            raise IntegrityError("duplicate key value violates unique constraint", params={}, orig=Exception())

        for inst in self.pending:
            if isinstance(inst, Document):
                # Check uniqueness constraint simulation
                existing = self.store.find_duplicate_document(
                    inst.user_id,
                    inst.content_hash,
                    inst.session_id,
                    inst.knowledge_base_id,
                )
                if existing:
                    raise IntegrityError("duplicate key value violates unique constraint", params={}, orig=Exception())

                if not inst.id:
                    inst.id = uuid.uuid4()
                now = datetime.now(timezone.utc)
                inst.created_at = now
                inst.updated_at = now
                self.store.documents[inst.id] = inst
            elif isinstance(inst, DocumentChunk):
                self.store.chunks.append(inst)
        self.pending.clear()

    async def rollback(self):
        self.pending.clear()

    async def refresh(self, instance):
        if not getattr(instance, "id", None):
            instance.id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        instance.created_at = getattr(instance, "created_at", now) or now
        instance.updated_at = now


@pytest.fixture
def mock_store():
    store = MockDBStore()
    return store


@pytest.fixture
def override_db(mock_store):
    async def _get_db():
        yield MockAsyncSession(mock_store)
    return _get_db


@pytest.mark.asyncio
async def test_duplicate_upload_same_user_and_session(mock_store, override_db):
    """Uploading the identical file twice under the same user and session returns duplicate response without re-processing."""
    user_id = uuid.UUID(USER_A["user_id"])
    session = mock_store.create_session(user_id=user_id)

    file_bytes = b"This is unique content for testing duplicate detection." * 10
    files = {"file": ("document.txt", file_bytes, "text/plain")}
    data = {"session_id": str(session.id)}

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg, \
         patch("app.routes.upload.get_supabase_client") as mock_supabase:

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = lambda: USER_A
            app.dependency_overrides[get_db] = override_db

            # 1. First upload
            res1 = await client.post("/api/v1/upload", data=data, files=files)
            assert res1.status_code == 202
            body1 = res1.json()
            assert body1["is_duplicate"] is False
            doc_id_1 = body1["document_id"]
            assert mock_bg.call_count == 1
            assert mock_supabase.return_value.storage.from_().upload.call_count == 1

            # 2. Second upload: identical bytes, same session
            files2 = {"file": ("document.txt", file_bytes, "text/plain")}
            res2 = await client.post("/api/v1/upload", data=data, files=files2)
            assert res2.status_code == 200
            body2 = res2.json()
            assert body2["is_duplicate"] is True
            assert body2["document_id"] == doc_id_1
            assert "Duplicate document detected" in body2["message"]

            # Background task should NOT be called again, storage upload NOT called again
            assert mock_bg.call_count == 1
            assert mock_supabase.return_value.storage.from_().upload.call_count == 1
            # Exactly 1 document in DB
            assert len(mock_store.documents) == 1


@pytest.mark.asyncio
async def test_duplicate_upload_different_filename_same_bytes(mock_store, override_db):
    """Deduplication is based on SHA-256 byte hash, not filename. Different filenames with same content are detected."""
    user_id = uuid.UUID(USER_A["user_id"])
    session = mock_store.create_session(user_id=user_id)

    file_bytes = b"Common body text across renamed files." * 10
    data = {"session_id": str(session.id)}

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg, \
         patch("app.routes.upload.get_supabase_client"):

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = lambda: USER_A
            app.dependency_overrides[get_db] = override_db

            # First upload as initial.txt
            res1 = await client.post("/api/v1/upload", data=data, files={"file": ("initial.txt", file_bytes, "text/plain")})
            assert res1.status_code == 202
            doc1_id = res1.json()["document_id"]

            # Second upload as renamed_final.txt
            res2 = await client.post("/api/v1/upload", data=data, files={"file": ("renamed_final.txt", file_bytes, "text/plain")})
            assert res2.status_code == 200
            body2 = res2.json()
            assert body2["is_duplicate"] is True
            assert body2["document_id"] == doc1_id
            assert mock_bg.call_count == 1


@pytest.mark.asyncio
async def test_identical_files_different_users_remain_independent(mock_store, override_db):
    """Different users uploading the same file content remain completely independent."""
    user_a_id = uuid.UUID(USER_A["user_id"])
    user_b_id = uuid.UUID(USER_B["user_id"])
    session_a = mock_store.create_session(user_id=user_a_id)
    session_b = mock_store.create_session(user_id=user_b_id)

    file_bytes = b"Shared knowledge base document content." * 10

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task"), \
         patch("app.routes.upload.get_supabase_client"):

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_db] = override_db

            # User A uploads
            app.dependency_overrides[get_current_user] = lambda: USER_A
            res_a = await client.post(
                "/api/v1/upload",
                data={"session_id": str(session_a.id)},
                files={"file": ("doc.txt", file_bytes, "text/plain")}
            )
            assert res_a.status_code == 202
            assert res_a.json()["is_duplicate"] is False
            doc_a_id = res_a.json()["document_id"]

            # User B uploads the same file bytes
            app.dependency_overrides[get_current_user] = lambda: USER_B
            res_b = await client.post(
                "/api/v1/upload",
                data={"session_id": str(session_b.id)},
                files={"file": ("doc.txt", file_bytes, "text/plain")}
            )
            assert res_b.status_code == 202
            assert res_b.json()["is_duplicate"] is False
            doc_b_id = res_b.json()["document_id"]

            assert doc_a_id != doc_b_id
            assert len(mock_store.documents) == 2


@pytest.mark.asyncio
async def test_identical_files_different_scopes_remain_independent(mock_store, override_db):
    """Same user uploading the same file to different sessions or knowledge bases creates distinct documents."""
    user_id = uuid.UUID(USER_A["user_id"])
    session_1 = mock_store.create_session(user_id=user_id)
    session_2 = mock_store.create_session(user_id=user_id)
    kb_1 = mock_store.create_kb(user_id=user_id)
    kb_2 = mock_store.create_kb(user_id=user_id)

    file_bytes = b"Multi-scope reusable text snippet." * 10

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task"), \
         patch("app.routes.upload.get_supabase_client"):

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = lambda: USER_A
            app.dependency_overrides[get_db] = override_db

            # 1. Upload to Session 1
            res_s1 = await client.post(
                "/api/v1/upload",
                data={"session_id": str(session_1.id)},
                files={"file": ("doc.txt", file_bytes, "text/plain")},
            )
            assert res_s1.status_code == 202
            assert res_s1.json()["is_duplicate"] is False

            # 2. Upload same bytes to Session 2 -> independent document
            res_s2 = await client.post(
                "/api/v1/upload",
                data={"session_id": str(session_2.id)},
                files={"file": ("doc.txt", file_bytes, "text/plain")},
            )
            assert res_s2.status_code == 202
            assert res_s2.json()["is_duplicate"] is False

            # 3. Upload same bytes to Knowledge Base 1 -> independent document
            res_kb1 = await client.post(
                f"/api/v1/knowledge-bases/{kb_1.id}/documents",
                files={"file": ("doc.txt", file_bytes, "text/plain")},
            )
            assert res_kb1.status_code == 202
            assert res_kb1.json()["is_duplicate"] is False

            # 4. Upload same bytes to Knowledge Base 1 again -> DUPLICATE in KB 1!
            res_kb1_dup = await client.post(
                f"/api/v1/knowledge-bases/{kb_1.id}/documents",
                files={"file": ("doc.txt", file_bytes, "text/plain")},
            )
            assert res_kb1_dup.status_code == 200
            assert res_kb1_dup.json()["is_duplicate"] is True
            assert res_kb1_dup.json()["document_id"] == res_kb1.json()["document_id"]

            # 5. Upload same bytes to Knowledge Base 2 -> independent document
            res_kb2 = await client.post(
                f"/api/v1/knowledge-bases/{kb_2.id}/documents",
                files={"file": ("doc.txt", file_bytes, "text/plain")},
            )
            assert res_kb2.status_code == 202
            assert res_kb2.json()["is_duplicate"] is False

            # We should have 4 unique documents total
            assert len(mock_store.documents) == 4


@pytest.mark.asyncio
async def test_different_content_same_filename(mock_store, override_db):
    """Uploading different content with the same filename creates a legitimate new document."""
    user_id = uuid.UUID(USER_A["user_id"])
    session = mock_store.create_session(user_id=user_id)
    data = {"session_id": str(session.id)}

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task"), \
         patch("app.routes.upload.get_supabase_client"):

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = lambda: USER_A
            app.dependency_overrides[get_db] = override_db

            res1 = await client.post(
                "/api/v1/upload",
                data=data,
                files={"file": ("report.txt", b"Version 1 content...", "text/plain")},
            )
            assert res1.status_code == 202
            assert res1.json()["is_duplicate"] is False

            res2 = await client.post(
                "/api/v1/upload",
                data=data,
                files={"file": ("report.txt", b"Version 2 content...", "text/plain")},
            )
            assert res2.status_code == 202
            assert res2.json()["is_duplicate"] is False
            assert res1.json()["document_id"] != res2.json()["document_id"]
            assert len(mock_store.documents) == 2


@pytest.mark.asyncio
async def test_empty_file_validation(mock_store, override_db):
    """Empty files are rejected with 400 Bad Request."""
    user_id = uuid.UUID(USER_A["user_id"])
    session = mock_store.create_session(user_id=user_id)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.dependency_overrides[get_current_user] = lambda: USER_A
        app.dependency_overrides[get_db] = override_db

        res = await client.post(
            "/api/v1/upload",
            data={"session_id": str(session.id)},
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert res.status_code == 400
        assert res.json()["detail"] == "File is empty."


@pytest.mark.asyncio
async def test_concurrent_upload_integrity_error_handled_gracefully(mock_store, override_db):
    """
    Race condition simulation:
    When two concurrent uploads pass the pre-check simultaneously,
    the second commit raises IntegrityError due to the unique constraint.
    The handler catches IntegrityError, cleans up storage, queries the winner,
    and returns 200 OK with is_duplicate=True.
    """
    user_id = uuid.UUID(USER_A["user_id"])
    session = mock_store.create_session(user_id=user_id)
    file_bytes = b"Concurrent race test content." * 10
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Pre-populate the winning document in the store so the rollback query finds it
    winning_doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        session_id=session.id,
        knowledge_base_id=None,
        filename="first_winner.txt",
        file_type="text/plain",
        file_path="winner_path",
        file_size=len(file_bytes),
        content_hash=content_hash,
        status="ready",
    )
    mock_store.documents[winning_doc.id] = winning_doc

    # Force commit to raise IntegrityError simulating constraint violation
    mock_store.fail_commit_with_integrity_error = True
    mock_store.race_condition_mode = True

    transport = httpx.ASGITransport(app=app)
    with patch("fastapi.BackgroundTasks.add_task") as mock_bg, \
         patch("app.routes.upload.get_supabase_client") as mock_supabase:

        mock_storage = MagicMock()
        mock_supabase.return_value = mock_storage

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            app.dependency_overrides[get_current_user] = lambda: USER_A
            app.dependency_overrides[get_db] = override_db

            res = await client.post(
                "/api/v1/upload",
                data={"session_id": str(session.id)},
                files={"file": ("second_race.txt", file_bytes, "text/plain")},
            )

            assert res.status_code == 200
            body = res.json()
            assert body["is_duplicate"] is True
            assert body["document_id"] == str(winning_doc.id)
            assert mock_bg.call_count == 0
            # Cleaned up uploaded transient file
            assert mock_storage.storage.from_().remove.call_count == 1
