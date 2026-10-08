import os
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test"
os.environ["GEMINI_API_KEY"] = "test"

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
