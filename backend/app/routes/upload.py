import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.session import ChatSession
from app.auth.dependencies import get_current_user
from app.config.database import get_db
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from ai.services.parser_service import parse_document
from ai.services.chunking_service import chunk_text
from ai.services.embedding_service import embedding_service

router = APIRouter(prefix="/upload", tags=["Document Upload"])


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

    # 1. Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty."
        )

    # 2. Extract text content
    try:
        raw_text = await parse_document(file_bytes, file.filename or "document.txt")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse document: {str(e)}",
        )

    if not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in the document.",
        )

    # 3. Save Document Record
    filename = file.filename or "document"
    storage_path = f"uploads/{session_id}/{filename}"  # Construct storage path string

    doc_record = Document(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        file_type=file.content_type or "text/plain",
        file_path=storage_path,  # <-- Added required non-null string
        file_size=len(file_bytes),
    )
    db.add(doc_record)
    await db.commit()
    await db.refresh(doc_record)

    # 4. Chunk raw text
    chunks = chunk_text(raw_text)

    # 5. Generate embeddings and store chunks
    for item in chunks:
        vector = embedding_service.generate_embedding(item["content"])
        chunk_record = DocumentChunk(
            document_id=doc_record.id,
            chunk_index=item["chunk_index"],
            content=item["content"],
            embedding=vector,
        )
        db.add(chunk_record)

    await db.commit()

    return {
        "message": "Document ingested successfully",
        "document_id": str(doc_record.id),
        "filename": doc_record.filename,
        "chunks_count": len(chunks),
    }