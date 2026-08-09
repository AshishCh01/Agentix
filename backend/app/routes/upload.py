import os
import uuid
import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.session import ChatSession
from app.auth.dependencies import get_current_user
from app.config.database import get_db
from app.config.settings import settings
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from ai.services.parser_service import parse_document
from ai.services.chunking_service import chunk_text
from ai.services.embedding_service import embedding_service

router = APIRouter(prefix="/upload", tags=["Document Upload"])

# Ensure local storage directory exists
UPLOAD_DIR = getattr(settings, "UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    session_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])

    # 1. Verify session exists and belongs to user
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    session_exists = result.scalar_one_or_none()

    if not session_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found. Please create a valid session first.",
        )

    # 2. Read file bytes and check max upload size limit
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

    # 3. Save physical file bytes to local storage directory
    session_upload_dir = os.path.join(UPLOAD_DIR, str(session_id))
    os.makedirs(session_upload_dir, exist_ok=True)
    file_path = os.path.join(session_upload_dir, f"{uuid.uuid4().hex}_{filename}")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    # 4. Extract text content per page
    try:
        pages_data = await parse_document(file_bytes, filename)
    except Exception as e:
        # Clean up saved file if parsing completely fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse document: {str(e)}",
        )

    has_content = any(page.get("text", "").strip() for page in pages_data)
    if not has_content:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in the document.",
        )

    # 5. Save Document Record with valid storage path
    doc_record = Document(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        file_type=file.content_type or "text/plain",
        file_path=file_path,
        file_size=len(file_bytes),
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    # 6. Chunk text per page and store chunks with page numbers
    all_chunks = []
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
                document_id=doc_record.id,
                chunk_index=global_chunk_index,
                page_number=page_num,
                content=item["content"],
                embedding=vector,
            )
            db.add(chunk_record)
            all_chunks.append(chunk_record)
            global_chunk_index += 1

    await db.commit()

    return {
        "message": "Document ingested successfully",
        "document_id": str(doc_record.id),
        "filename": doc_record.filename,
        "file_path": file_path,
        "chunks_count": len(all_chunks),
    }