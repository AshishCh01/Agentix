import uuid
import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, status, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError

from app.models.session import ChatSession
from app.auth.dependencies import get_current_user, get_supabase_client
from app.database.connection import AsyncSessionLocal, get_db
from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from ai.services.parser_service import parse_document
from ai.services.chunking_service import chunk_text
from ai.services.embedding_service import embedding_service, EmbeddingTask
from app.schemas.upload import DocumentUploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["Document Upload"])


async def process_document_background(
    doc_id: uuid.UUID,
    file_bytes: bytes,
    filename: str,
    storage_path: str,
):
    """
    Background worker with robust error recovery and explicit status logging.
    """
    logger.info(f"⏳ Starting background ingestion for document {doc_id} ({filename})")
    
    async with AsyncSessionLocal() as db:
        try:
            # 1. Parse document text
            logger.info(f"📄 Parsing text from document {doc_id}...")
            pages_data = await parse_document(file_bytes, filename)
            has_content = any(page.get("text", "").strip() for page in pages_data)

            if not has_content:
                logger.warning(f"⚠️ No readable text found in document {doc_id}. Marking as ready with 0 chunks.")
                res = await db.execute(select(Document).where(Document.id == doc_id))
                doc = res.scalar_one_or_none()
                if doc:
                    doc.status = "ready"
                    doc.error_message = "Document appears empty or contains no readable text."
                await db.commit()
                return

            # 2. Chunk text and prepare for embeddings
            logger.info(f"🧠 Chunking and preparing embeddings for document {doc_id}...")
            all_chunks = []
            global_chunk_index = 0
            
            for page_info in pages_data:
                page_num = page_info.get("page_number", 1)
                page_text = page_info.get("text", "")

                if not page_text.strip():
                    continue

                page_chunks = chunk_text(page_text)
                for item in page_chunks:
                    all_chunks.append({
                        "chunk_index": global_chunk_index,
                        "page_number": page_num,
                        "content": item["content"]
                    })
                    global_chunk_index += 1

            if not all_chunks:
                logger.warning(f"⚠️ No extractable chunks found for document {doc_id}.")
                res = await db.execute(select(Document).where(Document.id == doc_id))
                doc = res.scalar_one_or_none()
                if doc:
                    doc.status = "ready"
                    doc.error_message = "Document generated 0 chunks."
                await db.commit()
                return

            # Generate embeddings in batches of 32
            batch_size = 32
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                texts = [c["content"] for c in batch]
                embeddings = await embedding_service.generate_batch_embeddings(
                    texts, mode=EmbeddingTask.DOCUMENT, title=filename
                )
                if len(embeddings) != len(texts):
                    raise ValueError(f"Batch embedding returned {len(embeddings)} items, expected {len(texts)}.")

                for chunk_data, emb in zip(batch, embeddings):
                    chunk_record = DocumentChunk(
                        document_id=doc_id,
                        chunk_index=chunk_data["chunk_index"],
                        page_number=chunk_data["page_number"],
                        content=chunk_data["content"],
                        chunk_metadata={"filename": filename},
                        embedding=emb,
                    )
                    db.add(chunk_record)

            await db.flush()
            
            # Verify chunk counts match post-ingestion
            count_res = await db.execute(select(func.count()).where(DocumentChunk.document_id == doc_id))
            db_chunk_count = count_res.scalar() or 0
            if db_chunk_count != len(all_chunks):
                raise ValueError(f"Chunk count mismatch: expected {len(all_chunks)}, found {db_chunk_count} in database")

            # 3. Mark document status as 'ready'
            res = await db.execute(select(Document).where(Document.id == doc_id))
            doc = res.scalar_one_or_none()
            if doc:
                doc.status = "ready"
                doc.error_message = None

            await db.commit()
            logger.info(f"✅ Document {doc_id} processed successfully! ({global_chunk_index} chunks created)")

        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Ingestion error on document {doc_id}: {str(e)}", exc_info=True)

            # Open isolated DB session to record 'failed' status even if primary session failed
            try:
                async with AsyncSessionLocal() as err_db:
                    res = await err_db.execute(select(Document).where(Document.id == doc_id))
                    doc = res.scalar_one_or_none()
                    if doc:
                        doc.status = "failed"
                        doc.error_message = str(e)[:500]
                        await err_db.commit()
                        logger.info(f"⚠️ Document {doc_id} status updated to 'failed'")
            except Exception as commit_err:
                logger.error(f"❌ Failed to write error status for document {doc_id}: {commit_err}")


async def ingest_uploaded_file(
    background_tasks: BackgroundTasks,
    db: AsyncSession,
    user_id: uuid.UUID,
    file: UploadFile,
    *,
    session_id: Optional[uuid.UUID] = None,
    knowledge_base_id: Optional[uuid.UUID] = None,
) -> DocumentUploadResponse:
    """
    Shared ingestion path for both the session-scoped (/upload) and
    knowledge-base-scoped (/knowledge-bases/{id}/documents) upload
    endpoints: validates size, uploads bytes to Supabase Storage, creates
    the `processing` Document record (tagged with whichever of
    session_id/knowledge_base_id was provided), then offloads parsing +
    chunking + embedding to the same background worker either way.
    Caller is responsible for verifying ownership of session_id /
    knowledge_base_id before calling this.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty."
        )

    max_size_bytes = getattr(settings, "MAX_UPLOAD_SIZE_MB", 10) * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {getattr(settings, 'MAX_UPLOAD_SIZE_MB', 10)}MB.",
        )

    filename = file.filename or "document.txt"

    scope_segment = str(knowledge_base_id) if knowledge_base_id else str(session_id)
    storage_path = f"{user_id}/{scope_segment}/{uuid.uuid4().hex}_{filename}"

    # Upload file bytes to Supabase Storage Bucket
    supabase = get_supabase_client()
    try:
        bucket_name = getattr(settings, "SUPABASE_STORAGE_BUCKET", "documents")
        def _upload_to_supabase():
            return supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": file.content_type or "application/octet-stream"},
            )
        # Use asyncio.to_thread to prevent blocking the event loop
        await asyncio.to_thread(_upload_to_supabase)
    except Exception as e:
        logger.error(f"Supabase storage upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload file to external storage.",
        )

    # Save initial document record in 'processing' status
    try:
        doc_record = Document(
            user_id=user_id,
            session_id=session_id,
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            file_type=file.content_type or "text/plain",
            file_path=storage_path,
            file_size=len(file_bytes),
            status="processing",
        )
        db.add(doc_record)
        await db.commit()
        await db.refresh(doc_record)
    except SQLAlchemyError as db_err:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while saving document record.",
        )

    # Offload processing to background task
    background_tasks.add_task(
        process_document_background,
        doc_id=doc_record.id,
        file_bytes=file_bytes,
        filename=filename,
        storage_path=storage_path,
    )

    return DocumentUploadResponse(
        message="Document upload accepted. Processing in background.",
        document_id=doc_record.id,
        status="processing",
        filename=doc_record.filename,
    )


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    session_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])

    # Verify session exists and is owned by the caller
    try:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found.",
            )
    except SQLAlchemyError as db_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while verifying session.",
        )

    return await ingest_uploaded_file(
        background_tasks, db, user_id, file, session_id=session_id
    )