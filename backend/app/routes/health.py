from fastapi import APIRouter, status
from pydantic import BaseModel
from app.config.settings import settings

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    status: str
    project_name: str
    version: str


@router.get(
    "",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
)
async def health_check():
    return HealthStatus(
        status="online",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
    )