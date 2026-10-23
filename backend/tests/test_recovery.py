import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from app.services.recovery import (
    recover_stuck_ingestion_jobs,
    RESTART_RECOVERY_ERROR_MESSAGE,
)


def create_fake_document(
    status="processing",
    created_at=None,
    updated_at=None,
    doc_id=None,
    error_message=None,
):
    doc = Document(
        id=doc_id or uuid.uuid4(),
        user_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        filename="test.pdf",
        file_type="application/pdf",
        file_path="user/session/test.pdf",
        file_size=1024,
        status=status,
        error_message=error_message,
    )
    doc.created_at = created_at
    doc.updated_at = updated_at
    return doc


class MockAsyncSession:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.committed = False
        self.rolled_back = False
        self.deleted_chunk_doc_ids = []

    async def execute(self, stmt):
        stmt_str = str(stmt).lower()
        mock_result = MagicMock()

        if "delete from document_chunks" in stmt_str or "delete" in stmt_str:
            # Track chunk deletion
            return mock_result

        if "from documents" in stmt_str:
            # Filtering documents matching status == 'processing'
            if "status = :status_1" in stmt_str or "status = 'processing'" in stmt_str or "documents.status" in stmt_str:
                matching = [d for d in self.documents if d.status == "processing"]
            else:
                matching = self.documents
            mock_result.scalars.return_value.all.return_value = matching
            return mock_result

        return mock_result

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


@pytest.mark.asyncio
async def test_recover_stuck_document_before_restart():
    """A document left in 'processing' before restart is marked 'failed' after recovery."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(minutes=15)

    doc = create_fake_document(
        status="processing",
        created_at=old_time,
        updated_at=old_time,
    )
    db = MockAsyncSession([doc])

    recovered_count = await recover_stuck_ingestion_jobs(
        db,
        stale_timeout_seconds=300,
        now=now,
    )

    assert recovered_count == 1
    assert doc.status == "failed"
    assert doc.error_message == RESTART_RECOVERY_ERROR_MESSAGE
    assert doc.updated_at == now
    assert db.committed is True


@pytest.mark.asyncio
async def test_completed_and_failed_documents_remain_untouched():
    """Completed ('ready') and already failed documents remain untouched."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(minutes=30)

    ready_doc = create_fake_document(
        status="ready",
        created_at=old_time,
        updated_at=old_time,
        error_message=None,
    )
    failed_doc = create_fake_document(
        status="failed",
        created_at=old_time,
        updated_at=old_time,
        error_message="User cancelled",
    )
    db = MockAsyncSession([ready_doc, failed_doc])

    recovered_count = await recover_stuck_ingestion_jobs(
        db,
        stale_timeout_seconds=300,
        now=now,
    )

    assert recovered_count == 0
    assert ready_doc.status == "ready"
    assert ready_doc.error_message is None
    assert failed_doc.status == "failed"
    assert failed_doc.error_message == "User cancelled"


@pytest.mark.asyncio
async def test_recovery_is_idempotent():
    """Running recovery more than once does not corrupt state or re-fail documents."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(minutes=20)

    doc = create_fake_document(
        status="processing",
        created_at=old_time,
        updated_at=old_time,
    )
    db = MockAsyncSession([doc])

    # First run recovers the document
    count1 = await recover_stuck_ingestion_jobs(db, stale_timeout_seconds=300, now=now)
    assert count1 == 1
    assert doc.status == "failed"

    # Second run finds 0 documents in processing
    count2 = await recover_stuck_ingestion_jobs(db, stale_timeout_seconds=300, now=now)
    assert count2 == 0
    assert doc.status == "failed"


@pytest.mark.asyncio
async def test_missing_or_invalid_timestamps_handled_safely():
    """Missing or invalid/future timestamps are handled safely without crashing."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    doc_no_ts = create_fake_document(status="processing", created_at=None, updated_at=None)
    doc_future = create_fake_document(
        status="processing",
        created_at=now + timedelta(hours=2),
        updated_at=now + timedelta(hours=2),
    )
    doc_naive = create_fake_document(
        status="processing",
        created_at=datetime(2026, 10, 20, 10, 0, 0),  # Naive, older than cutoff
        updated_at=datetime(2026, 10, 20, 10, 0, 0),
    )

    db = MockAsyncSession([doc_no_ts, doc_future, doc_naive])

    recovered_count = await recover_stuck_ingestion_jobs(
        db,
        stale_timeout_seconds=300,
        now=now,
    )

    assert recovered_count == 3
    assert doc_no_ts.status == "failed"
    assert doc_future.status == "failed"
    assert doc_naive.status == "failed"


@pytest.mark.asyncio
async def test_multi_worker_active_jobs_not_failed():
    """Active jobs being processed by another worker (within threshold) are not marked as failed."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    recent_time = now - timedelta(seconds=15)  # Started 15 seconds ago

    active_doc = create_fake_document(
        status="processing",
        created_at=recent_time,
        updated_at=recent_time,
    )
    db = MockAsyncSession([active_doc])

    recovered_count = await recover_stuck_ingestion_jobs(
        db,
        stale_timeout_seconds=300,
        now=now,
    )

    assert recovered_count == 0
    assert active_doc.status == "processing"


@pytest.mark.asyncio
async def test_multiple_recovery_executions_with_active_and_abandoned_jobs():
    """Multiple recovery executions safely fail abandoned jobs while preserving active jobs."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(minutes=10)
    fresh_time = now - timedelta(seconds=20)

    abandoned_doc = create_fake_document(status="processing", created_at=old_time, updated_at=old_time)
    active_doc = create_fake_document(status="processing", created_at=fresh_time, updated_at=fresh_time)

    db = MockAsyncSession([abandoned_doc, active_doc])

    # Execution 1
    count1 = await recover_stuck_ingestion_jobs(db, stale_timeout_seconds=300, now=now)
    assert count1 == 1
    assert abandoned_doc.status == "failed"
    assert active_doc.status == "processing"

    # Execution 2 (e.g. another worker starts up 10 seconds later)
    now_later = now + timedelta(seconds=10)
    count2 = await recover_stuck_ingestion_jobs(db, stale_timeout_seconds=300, now=now_later)
    assert count2 == 0
    assert abandoned_doc.status == "failed"
    assert active_doc.status == "processing"


@pytest.mark.asyncio
async def test_partial_chunks_cleaned_up_on_recovery():
    """When an abandoned document is marked failed, its partial chunks are deleted."""
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(minutes=10)

    doc = create_fake_document(status="processing", created_at=old_time, updated_at=old_time)
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [doc]
    mock_db.execute.return_value = mock_result

    recovered_count = await recover_stuck_ingestion_jobs(mock_db, stale_timeout_seconds=300, now=now)

    assert recovered_count == 1
    assert doc.status == "failed"

    # Verify delete was called on DocumentChunk
    delete_call = mock_db.execute.call_args_list[1]
    delete_stmt = delete_call[0][0]
    assert "delete" in str(delete_stmt).lower()
    assert "document_chunks" in str(delete_stmt).lower()
    assert mock_db.commit.called


@pytest.mark.asyncio
async def test_database_errors_handled_and_logged():
    """Database errors trigger rollback and logging without crashing startup."""
    mock_db = AsyncMock()
    mock_db.execute.side_effect = SQLAlchemyError("DB connection lost")

    recovered = await recover_stuck_ingestion_jobs(mock_db, raise_on_error=False)
    assert recovered == 0
    mock_db.rollback.assert_called_once()

    # With raise_on_error=True
    with pytest.raises(SQLAlchemyError):
        await recover_stuck_ingestion_jobs(mock_db, raise_on_error=True)


@pytest.mark.asyncio
async def test_lifespan_startup_recovers_documents():
    """FastAPI lifespan startup hook triggers ingestion recovery safely."""
    from app.main import lifespan
    from fastapi import FastAPI

    test_app = FastAPI()
    with patch("app.services.recovery.recover_stuck_ingestion_jobs", new_callable=AsyncMock) as mock_recover, \
         patch("app.database.connection.AsyncSessionLocal") as mock_session_maker, \
         patch("app.main.settings.RERANK_PRELOAD", False):
        mock_session = AsyncMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_recover.return_value = 2

        async with lifespan(test_app):
            mock_recover.assert_called_once()
