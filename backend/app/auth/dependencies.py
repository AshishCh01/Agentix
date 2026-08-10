import jwt
import time
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client
from app.config.settings import settings

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_URL or SUPABASE_KEY is missing from environment variables.",
            )
        _supabase_client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )
    return _supabase_client


security = HTTPBearer()


from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database import crud

_last_sync_time: dict[str, float] = {}
SYNC_INTERVAL = 300  # 5 minutes

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Validates tokens locally via PyJWT when SUPABASE_JWT_SECRET is set to eliminate network latency,
    falling back to remote Supabase Auth network calls if secret is absent or decoding fails.
    """
    token = credentials.credentials
    user_dict = None

    # 1. Local JWT Verification (Instant, zero network overhead)
    if settings.SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            uid_str = str(payload.get("sub") or payload.get("id"))
            if not uid_str:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload: missing sub/id",
                )
            user_dict = {
                "user_id": uid_str,
                "id": uid_str,
                "sub": uid_str,
                "email": payload.get("email", ""),
                "full_name": payload.get("user_metadata", {}).get("full_name") if payload.get("user_metadata") else None,
                "avatar_url": payload.get("user_metadata", {}).get("avatar_url") if payload.get("user_metadata") else None,
                "role": payload.get("role", "authenticated"),
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.PyJWTError:
            # Token is invalid structurally; do not fallback to avoid spamming Supabase API
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token format",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 2. Remote Verification Fallback (if secret not configured)
    if not user_dict:
        supabase = get_supabase_client()
        try:
            user_response = supabase.auth.get_user(token)

            if not user_response or not user_response.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session",
                )

            user = user_response.user
            uid_str = str(user.id)

            user_dict = {
                "user_id": uid_str,
                "id": uid_str,
                "sub": uid_str,
                "email": user.email or "",
                "full_name": user.user_metadata.get("full_name") if user.user_metadata else None,
                "avatar_url": user.user_metadata.get("avatar_url") if user.user_metadata else None,
                "role": getattr(user, "role", "authenticated"),
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Supabase Auth error: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 3. Synchronize User to Local DB (Cached every 5 mins)
    now = time.time()
    if now - _last_sync_time.get(user_dict["user_id"], 0) > SYNC_INTERVAL:
        await crud.sync_user(
            db=db,
            user_id=user_dict["user_id"],
            email=user_dict["email"],
            full_name=user_dict.get("full_name"),
            avatar_url=user_dict.get("avatar_url"),
        )
        _last_sync_time[user_dict["user_id"]] = now

    return user_dict