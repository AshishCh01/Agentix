import uuid
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.session import ChatSession
from app.auth.dependencies import get_current_user, get_supabase_client
from app.config.database import get_db
from app.database.connection import AsyncSessionLocal
from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from ai.services.parser_service import parse_document
from ai.services.chunking_service import chunk_text
from ai.services.embedding_service import embedding_service

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
                raise ValueError("No readable text found in document.")

            # 2. Chunk text and generate embeddings
            logger.info(f"🧠 Generating embeddings for document {doc_id}...")
            global_chunk_index = 0
            for page_info in pages_data:
                page_num = page_info.get("page_number", 1)
                page_text = page_info.get("text", "")

                if not page_text.strip():
                    continue

                page_chunks = chunk_text(page_text)
                for item in page_chunks:
                    vector = embedding_service.generate_embedding(item["content"])
                    chunk_record = DocumentChunk(
                        document_id=doc_id,
                        chunk_index=global_chunk_index,
                        page_number=page_num,
                        content=item["content"],
                        chunk_metadata={},
                        embedding=vector,
                    )
                    db.add(chunk_record)
                    global_chunk_index += 1

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


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    session_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])

    # 1. Verify session exists
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

    # 2. Read bytes and check size
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

    #storage_path = f"{session_id}/{uuid.uuid4().hex}_{filename}"
    storage_path = f"{user_id}/{session_id}/{uuid.uuid4().hex}_{filename}"

    # 3. Upload file bytes to Supabase Storage Bucket
    supabase = get_supabase_client()
    try:
        bucket_name = getattr(settings, "SUPABASE_STORAGE_BUCKET", "documents")
        supabase.storage.from_(bucket_name).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": file.content_type or "application/octet-stream"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to storage: {str(e)}",
        )

    # 4. Save initial document record in 'processing' status
    doc_record = Document(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        file_type=file.content_type or "text/plain",
        file_path=storage_path,
        file_size=len(file_bytes),
        status="processing",
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    # 5. Offload processing to background task
    background_tasks.add_task(
        process_document_background,
        doc_id=doc_record.id,
        file_bytes=file_bytes,
        filename=filename,
        storage_path=storage_path,
    )

    return {
        "message": "Document upload accepted. Processing in background.",
        "document_id": str(doc_record.id),
        "status": "processing",
        "filename": doc_record.filename,
    }