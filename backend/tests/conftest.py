import pytest
from ai.services.llm_service import llm_service
from app.main import embedding_service
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
