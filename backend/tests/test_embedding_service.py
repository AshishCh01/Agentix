import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from google.genai.errors import APIError
from app.config.settings import settings
from ai.services.embedding_service import embedding_service, EmbeddingTask
import math

class MockResponse:
    def __init__(self, values):
        self.values = values

class MockEmbeddings:
    def __init__(self, embeddings_list):
        self.embeddings = [MockResponse(v) for v in embeddings_list]

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock()
    return client

@pytest.fixture(autouse=True)
def setup_embedding_service(mock_client):
    embedding_service._client = mock_client
    # Reset concurrency before each test just in case
    settings.EMBED_CONCURRENCY = 4
    yield
    embedding_service._client = None

@pytest.mark.asyncio
async def test_normal_batch_succeeds(mock_client):
    mock_client.aio.models.embed_content.return_value = MockEmbeddings([[0.1] * 2000])
    texts = ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]
    embeddings = await embedding_service.generate_batch_embeddings(texts)
    
    assert len(embeddings) == 4
    assert mock_client.aio.models.embed_content.call_count == 4
    for emb in embeddings:
        assert len(emb) == 2000
        assert emb[0] == 0.1

@pytest.mark.asyncio
async def test_embedding_order_preserved(mock_client):
    # Make the mock return different embeddings based on the content
    async def mock_embed_content(*args, **kwargs):
        content = kwargs.get("contents")[0]
        # Just use length as a dummy differentiator for test
        val = float(len(content))
        return MockEmbeddings([[val] * 2000])
        
    mock_client.aio.models.embed_content.side_effect = mock_embed_content
    
    texts = ["a", "ab", "abc"]
    embeddings = await embedding_service.generate_batch_embeddings(texts)
    
    assert len(embeddings) == 3
    assert embeddings[0][0] == float(len("title: none | text: a"))
    assert embeddings[1][0] == float(len("title: none | text: ab"))
    assert embeddings[2][0] == float(len("title: none | text: abc"))

@pytest.mark.asyncio
async def test_empty_input_rejected():
    with pytest.raises(ValueError, match="Empty input text"):
        await embedding_service.generate_embedding("   ")

@pytest.mark.asyncio
async def test_wrong_dimension_rejected(mock_client):
    mock_client.aio.models.embed_content.return_value = MockEmbeddings([[0.1] * 1999])
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await embedding_service.generate_embedding("valid text")

@pytest.mark.asyncio
async def test_non_finite_rejected(mock_client):
    mock_client.aio.models.embed_content.return_value = MockEmbeddings([[math.inf] + [0.1] * 1999])
    with pytest.raises(ValueError, match="Invalid embedding value found"):
        await embedding_service.generate_embedding("valid text")

    mock_client.aio.models.embed_content.return_value = MockEmbeddings([[math.nan] + [0.1] * 1999])
    with pytest.raises(ValueError, match="Invalid embedding value found"):
        await embedding_service.generate_embedding("valid text")

@pytest.mark.asyncio
async def test_retry_transient_failures_and_429(mock_client):
    # Fail twice with 429, succeed on 3rd
    err = Exception("429 ResourceExhausted")
    mock_client.aio.models.embed_content.side_effect = [err, err, MockEmbeddings([[0.5] * 2000])]
    
    emb = await embedding_service._embed_with_retry("text", EmbeddingTask.DOCUMENT, None, asyncio.Semaphore(1))
    assert emb[0] == 0.5
    assert mock_client.aio.models.embed_content.call_count == 3

@pytest.mark.asyncio
async def test_no_retry_4xx_non_429(mock_client):
    class FakeError(Exception):
        code = 400
    
    err = FakeError("400 Bad Request")
    mock_client.aio.models.embed_content.side_effect = [err, MockEmbeddings([[0.5] * 2000])]
    
    with pytest.raises(FakeError):
        await embedding_service._embed_with_retry("text", EmbeddingTask.DOCUMENT, None, asyncio.Semaphore(1))
        
    assert mock_client.aio.models.embed_content.call_count == 1

@pytest.mark.asyncio
async def test_concurrency_limit_respected(mock_client):
    # Track concurrent executions
    concurrent_calls = 0
    max_concurrent = 0
    
    async def mock_embed_content(*args, **kwargs):
        nonlocal concurrent_calls, max_concurrent
        concurrent_calls += 1
        max_concurrent = max(max_concurrent, concurrent_calls)
        await asyncio.sleep(0.01) # Hold the lock slightly
        concurrent_calls -= 1
        return MockEmbeddings([[0.1] * 2000])
        
    mock_client.aio.models.embed_content.side_effect = mock_embed_content
    settings.EMBED_CONCURRENCY = 4
    
    texts = [f"text {i}" for i in range(10)]
    await embedding_service.generate_batch_embeddings(texts)
    
    # max_concurrent should not exceed 4
    assert max_concurrent <= 4
    assert mock_client.aio.models.embed_content.call_count == 10
