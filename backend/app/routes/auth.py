from fastapi import APIRouter, HTTPException, status
from app.schemas.auth import UserLogin, UserSignup, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: UserSignup):
    """
    User signup endpoint.
    """
    return {"message": "User signup endpoint ready", "email": payload.email}


@router.post("/login", status_code=status.HTTP_200_OK)
async def login(payload: UserLogin):
    """
    User login endpoint.
    """
    return {"message": "User login endpoint ready", "email": payload.email}