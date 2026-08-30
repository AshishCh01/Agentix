import re
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic RAG System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    HF_TOKEN: str = ""

    # Ingestion & File Storage Settings
    SUPABASE_STORAGE_BUCKET: str = "documents"  # Name of your Supabase storage bucket
    MAX_UPLOAD_SIZE_MB: int = 10

    # Optimization Controls
    SKIP_REFLECTION: bool = False  # Skip reflection for faster responses

    # Supabase Credentials
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_JWT_SECRET: str = ""

    # Database Settings
    DATABASE_URL: str = ""

    def _database_url_with_driver(self, driver: str) -> str:
        """
        DATABASE_URL rewritten to use the given driver, regardless of which
        scheme/driver (postgres://, postgresql://, postgresql+psycopg2://,
        postgresql+asyncpg://, etc.) is actually configured in the
        environment. This is the single place DATABASE_URL is parsed, so the
        app runtime and Alembic can each get the driver they need without
        DATABASE_URL itself being touched.
        """
        return re.sub(r"^postgres(ql)?(\+\w+)?://", f"postgresql+{driver}://", self.DATABASE_URL or "")

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """
        DATABASE_URL normalized to the asyncpg driver, for the FastAPI/
        SQLAlchemy async engine (app/database/connection.py). asyncpg is the
        only driver that works with SQLAlchemy's async engine API.
        """
        return self._database_url_with_driver("asyncpg")

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """
        DATABASE_URL normalized to the psycopg2 driver, for Alembic
        (alembic/env.py), which runs migrations synchronously.
        """
        return self._database_url_with_driver("psycopg2")

    # LLM Settings
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-3.5-flash"  # Default LLM model for responses

    # Embedding Settings
    # gemini-embedding-2's native output is 3072 dims, but pgvector's HNSW
    # index has a hard cap of 2000 dims -- truncated here via
    # output_dimensionality so the HNSW index stays usable.
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIMENSIONS: int = 2000

    # CORS Configurations
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()