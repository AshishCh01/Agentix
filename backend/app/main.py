from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.routes import chat, health, knowledge_base, search, session, upload

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN


@asynccontextmanager
async def lifespan(app: FastAPI):
    import time
    import psutil
    import asyncio
    
    # --- STARTUP LOGIC ---
    if not settings.SUPABASE_URL:
        print(
            "[WARN] SUPABASE_URL is not set. Every authenticated request will "
            "fall back to a remote Supabase Auth network call instead of fast local "
            "JWT verification via Supabase's JWKS endpoint, adding latency to every "
            "request."
        )

    # Dependency Checks
    optional_deps = ["ddgs", "openpyxl", "xlrd"]
    if settings.RERANK_PROVIDER == "local":
        optional_deps.append("sentence_transformers")
    
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            print(f"[WARN] Dependency '{dep}' is missing. Some features may not work or might fail.")

    # Reranker Warmup
    # Note: Each uvicorn worker process loads its own copy of the model. 
    # Using --reload will load it again after each reload.
    if settings.RERANK_PROVIDER == "local" and settings.RERANK_PRELOAD:
        print("[*] Pre-loading and warming up local reranker...")
        from ai.services.embedding_service import embedding_service
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        t0 = time.time()
        try:
            await asyncio.to_thread(embedding_service.warmup_reranker)
            t1 = time.time()
            mem_after = process.memory_info().rss / 1024 / 1024
            print(f"[OK] Reranker warmed up in {t1 - t0:.2f}s. Memory before: {mem_before:.2f}MB, after: {mem_after:.2f}MB")
        except Exception as e:
            print(f"[WARN] Failed to warm up reranker: {e}")

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
app.include_router(knowledge_base.router, prefix=settings.API_V1_STR)
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