from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic RAG System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Supabase Credentials
    SUPABASE_URL: str =""
    SUPABASE_KEY: str =""

    # Database Settings
    DATABASE_URL: str =""

    # LLM Settings
    GEMINI_API_KEY: str = ""

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