from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.routes import auth, chat, health, search, session, upload

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
# Import EmbeddingService targeting ai/services/embedding_service.py
from ai.services.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("⚡ Pre-loading PyTorch and embedding model into RAM...")
    try:
        service = EmbeddingService()
        
        # Use getattr to look up methods dynamically and bypass Pylance static checks
        embed_fn = (
            getattr(service, "embed_query", None)
            or getattr(service, "embed_text", None)
            or getattr(service, "generate_embedding", None)
            or getattr(service, "get_embeddings", None)
        )
        
        if callable(embed_fn):
            embed_fn("warmup")
            
        print("✅ Embedding model weights loaded into RAM successfully.")
    except Exception as e:
        print(f"⚠️ Warning during embedding model pre-load: {e}")

    yield  # Application handles requests here

    # --- SHUTDOWN LOGIC ---
    print("🛑 Shutting down application...")


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
app.include_router(auth.router, prefix=settings.API_V1_STR)
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