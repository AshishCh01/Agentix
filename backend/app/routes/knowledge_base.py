import uuid
from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.database import crud
from app.routes.upload import ingest_uploaded_file
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from app.schemas.upload import DocumentListResponse, DocumentResponse, DocumentUploadResponse

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])


def _to_response(kb, document_count: int = 0) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        user_id=kb.user_id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        document_count=document_count,
    )


async def _get_owned_knowledge_base(db: AsyncSession, kb_id: uuid.UUID, user_id: uuid.UUID):
    """Fetches a knowledge base and 404s if it doesn't exist or isn't owned by user_id."""
    kb = await crud.get_knowledge_base(db, knowledge_base_id=kb_id, user_id=user_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )
    return kb


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    kb_in: KnowledgeBaseCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        kb = await crud.create_knowledge_base(
            db, user_id=user_id, name=kb_in.name, description=kb_in.description
        )
        return _to_response(kb, document_count=0)
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while creating knowledge base.",
        )


@router.get("", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    limit: int = Query(50, ge=1, le=200, description="Max number of knowledge bases to return"),
    offset: int = Query(0, ge=0, description="Number of knowledge bases to skip"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        rows = await crud.get_user_knowledge_bases(db, user_id=user_id, limit=limit, offset=offset)
        return [_to_response(kb, document_count=count) for kb, count in rows]
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while fetching knowledge bases.",
        )


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        kb = await _get_owned_knowledge_base(db, kb_id, user_id)
        count = await crud.get_knowledge_base_document_count(db, kb_id)
        return _to_response(kb, document_count=count)
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while fetching knowledge base.",
        )


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: uuid.UUID,
    kb_in: KnowledgeBaseUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        # Ownership check first so a non-owner gets a 404 rather than a
        # successful no-op update.
        await _get_owned_knowledge_base(db, kb_id, user_id)
        updated = await crud.update_knowledge_base(
            db, knowledge_base_id=kb_id, user_id=user_id,
            name=kb_in.name, description=kb_in.description,
        )
        count = await crud.get_knowledge_base_document_count(db, kb_id)
        return _to_response(updated, document_count=count)
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while updating knowledge base.",
        )


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        success = await crud.delete_knowledge_base(db, knowledge_base_id=kb_id, user_id=user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
            )
        return None
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while deleting knowledge base.",
        )


@router.get("/{kb_id}/documents", response_model=DocumentListResponse)
async def list_knowledge_base_documents(
    kb_id: uuid.UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        await _get_owned_knowledge_base(db, kb_id, user_id)
        docs = await crud.get_knowledge_base_documents(db, kb_id, limit=limit, offset=offset)
        return DocumentListResponse(documents=[DocumentResponse.model_validate(d) for d in docs])
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while fetching knowledge base documents.",
        )


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_knowledge_base_document(
    kb_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    await _get_owned_knowledge_base(db, kb_id, user_id)
    return await ingest_uploaded_file(
        background_tasks, db, user_id, file, knowledge_base_id=kb_id
    )


@router.delete("/{kb_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base_document(
    kb_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = uuid.UUID(current_user["user_id"])
    try:
        await _get_owned_knowledge_base(db, kb_id, user_id)
        doc = await crud.get_knowledge_base_document(db, kb_id, document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )
        await crud.delete_document(db, document_id)
        return None
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error while deleting document.",
        )
