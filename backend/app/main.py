from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.routes import chat, health, search, session, upload

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    if not settings.SUPABASE_JWT_SECRET:
        print(
            "[WARN] SUPABASE_JWT_SECRET is not set. Every authenticated request will "
            "fall back to a remote Supabase Auth network call instead of fast local "
            "JWT verification, adding latency to every request. Set SUPABASE_JWT_SECRET "
            "in the environment to enable local verification."
        )

    # No embedding model warmup: embeddings are generated via Gemini's async
    # API (ai/services/embedding_service.py), not a locally-loaded model, so
    # there are no weights to pre-load into RAM at startup.
    print("[OK] Startup checks complete.")

    yield  # Application handles requests here

    # --- SHUTDOWN LOGIC ---
    print("[--] Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(session.router, prefix=settings.API_V1_STR)
app.include_router(upload.router, prefix=settings.API_V1_STR)
app.include_router(search.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }