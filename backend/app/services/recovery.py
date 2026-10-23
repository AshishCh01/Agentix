import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunks import DocumentChunk

logger = logging.getLogger(__name__)

RESTART_RECOVERY_ERROR_MESSAGE = (
    "Ingestion interrupted by server restart or process termination. "
    "Please re-upload the document."
)


async def recover_stuck_ingestion_jobs(
    db: AsyncSession,
    *,
    stale_timeout_seconds: Optional[int] = None,
    now: Optional[datetime] = None,
    raise_on_error: bool = False,
) -> int:
    """
    Recovers documents left in 'processing' status by interrupted ingestion jobs.
    
    In a BackgroundTasks-based architecture, any process shutdown, restart, or worker crash
    terminates in-memory background ingestion tasks without executing error handlers.
    This leaves documents permanently stuck in 'processing' unless recovered on startup.
    
    This function is idempotent and safe for multi-worker setups:
    - Only queries documents whose status is 'processing'.
    - If a document was created or updated recently (within stale_timeout_seconds),
      it is presumed to be actively running on another worker and is NOT touched.
    - If a document has an outdated timestamp (older than cutoff), missing timestamp (None),
      or invalid timestamp, it is considered abandoned.
    - When abandoned jobs are failed, any partial chunks in document_chunks are cleaned up
      atomically with the status update to avoid inconsistent or duplicate chunk state.
    - Handles database errors safely with rollback and logging.
    
    Returns the number of abandoned documents recovered.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if stale_timeout_seconds is None:
        timeout = getattr(settings, "INGESTION_STALE_TIMEOUT_SECONDS", 300)
    else:
        timeout = stale_timeout_seconds

    cutoff = now - timedelta(seconds=timeout)

    try:
        # Fetch all documents currently in 'processing' status
        stmt = select(Document).where(Document.status == "processing")
        result = await db.execute(stmt)
        processing_docs: List[Document] = list(result.scalars().all())

        if not processing_docs:
            logger.debug("Ingestion recovery check: no processing documents found.")
            return 0

        abandoned_docs: List[Document] = []
        for doc in processing_docs:
            ref_time = getattr(doc, "updated_at", None) or getattr(doc, "created_at", None)

            is_stale = False
            if ref_time is None:
                # Missing timestamp -> cannot verify active execution, treat as abandoned
                is_stale = True
            else:
                # Ensure timezone-aware for comparison
                if ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=timezone.utc)

                # Check if timestamp is in the future by more than 5 minutes (invalid / clock drift)
                if ref_time > now + timedelta(minutes=5):
                    is_stale = True
                elif ref_time <= cutoff:
                    is_stale = True

            if is_stale:
                abandoned_docs.append(doc)

        if not abandoned_docs:
            logger.info(
                f"Ingestion recovery check: {len(processing_docs)} processing document(s) detected, "
                "all active within recent window. None marked as failed."
            )
            return 0

        # Mark abandoned documents as failed and clean up partial chunks
        doc_ids = [doc.id for doc in abandoned_docs]

        # 1. Delete orphan/partial chunks for all abandoned documents
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids))
        )

        # 2. Update status and error_message
        for doc in abandoned_docs:
            doc.status = "failed"
            doc.error_message = RESTART_RECOVERY_ERROR_MESSAGE
            doc.updated_at = now

        await db.commit()

        logger.warning(
            f"⚠️ Recovered {len(abandoned_docs)} abandoned ingestion job(s) from restart/interruption: "
            f"document IDs: {doc_ids}"
        )
        return len(abandoned_docs)

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"❌ Database error during ingestion recovery: {e}", exc_info=True)
        if raise_on_error:
            raise
        return 0
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Unexpected error during ingestion recovery: {e}", exc_info=True)
        if raise_on_error:
            raise
        return 0
