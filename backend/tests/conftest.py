import os

# Set dummy environment variables to prevent crashes during test collection
# when no .env is present. setdefault ensures we don't override real variables
# provided by CI or a local .env (if os.environ was pre-populated).
import os
from dotenv import dotenv_values

env_vars = dotenv_values(".env")

if "DATABASE_URL" not in os.environ and not env_vars.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"

if "SUPABASE_URL" not in os.environ and not env_vars.get("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = "https://dummy.supabase.co"

if "SUPABASE_KEY" not in os.environ and not env_vars.get("SUPABASE_KEY"):
    os.environ["SUPABASE_KEY"] = "dummy_key"

if "GEMINI_API_KEY" not in os.environ and not env_vars.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "dummy_gemini"

import pytest
from ai.services.llm_service import llm_service
import asyncio

@pytest.fixture(autouse=True)
def clear_caches():
    """
    Clear singletons' cached models before each test to prevent 
    'Event loop is closed' errors when tests run in separate event loops.
    """
    if hasattr(llm_service, "_models"):
        llm_service._models.clear()
        
    yield
