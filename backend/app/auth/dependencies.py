import jwt
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


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Validates tokens locally via PyJWT when SUPABASE_JWT_SECRET is set to eliminate network latency,
    falling back to remote Supabase Auth network calls if secret is absent or decoding fails.
    """
    token = credentials.credentials

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
            return {
                "user_id": uid_str,
                "id": uid_str,
                "sub": uid_str,
                "email": payload.get("email", ""),
                "role": payload.get("role", "authenticated"),
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.PyJWTError:
            pass  # Fall through to remote validation if token verification fails locally

    # 2. Remote Verification Fallback
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

        return {
            "user_id": uid_str,
            "id": uid_str,
            "sub": uid_str,
            "email": user.email or "",
            "role": getattr(user, "role", "authenticated"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Supabase Auth error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )