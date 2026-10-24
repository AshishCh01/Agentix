import pytest
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from app.routes.upload import process_document_background
from app.models.document import Document
from app.models.document_chunks import DocumentChunk

@pytest.fixture
def mock_embedding_service():
    with patch("app.routes.upload.embedding_service") as mock_es:
        async def generate_batch_embeddings(texts, **kwargs):
            return [[float(len(t))] * 2000 for t in texts]
        mock_es.generate_batch_embeddings = AsyncMock(side_effect=generate_batch_embeddings)
        yield mock_es

@pytest.fixture
def mock_parser_service():
    with patch("app.routes.upload.parse_document") as mock_parse:
        yield mock_parse

@pytest.fixture
def mock_async_session_local():
    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker:
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session
        yield mock_session

@pytest.mark.asyncio
async def test_process_70_chunks(mock_parser_service, mock_embedding_service, mock_async_session_local):
    # Setup mock parser to return 70 chunks worth of text
    # Let's say 1 chunk is typically ~500 chars depending on chunking_service
    # We will just pass 70 pages of distinct text so we get exactly 70 chunks.
    # But wait, chunking is real. Let's just mock `chunk_text` to be simple.
    with patch("app.routes.upload.chunk_text") as mock_chunk:
        mock_chunk.side_effect = lambda text: [{"content": text}]
        
        mock_parser_service.return_value = [{"page_number": i, "text": f"distinct text {i}"} for i in range(70)]
        
        doc_id = uuid.uuid4()
        
        # We need `db.execute` to return a mock document for `scalar_one_or_none()` and `scalar()` (for count)
        mock_doc = MagicMock()
        
        async def mock_execute(stmt):
            mock_res = MagicMock()
            if "count(" in str(stmt).lower():
                mock_res.scalar.return_value = 70
            else:
                mock_res.scalar_one_or_none.return_value = mock_doc
            return mock_res
            
        mock_async_session_local.execute.side_effect = mock_execute
        
        await process_document_background(doc_id, b"dummy", "dummy.pdf", "path/to/dummy.pdf")
        
        # Verify exactly 70 chunks added
        add_calls = mock_async_session_local.add.call_args_list
        assert len(add_calls) == 70
        
        for i, call in enumerate(add_calls):
            chunk = call[0][0]
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == doc_id
            assert chunk.chunk_index == i
            assert chunk.content == f"distinct text {i}"
            assert chunk.embedding[0] == float(len(f"distinct text {i}"))
            
        # Verify Document status updated to ready
        assert mock_doc.status == "ready"
        assert mock_async_session_local.commit.called

@pytest.mark.asyncio
async def test_failed_upload_leaves_no_chunks(mock_parser_service, mock_embedding_service, mock_async_session_local):
    with patch("app.routes.upload.chunk_text") as mock_chunk:
        mock_chunk.side_effect = lambda text: [{"content": text}]
        mock_parser_service.return_value = [{"page_number": i, "text": f"distinct text {i}"} for i in range(10)]
        
        # Simulate embedding service failure
        mock_embedding_service.generate_batch_embeddings.side_effect = ValueError("Simulated embedding failure")
        
        doc_id = uuid.uuid4()
        
        mock_doc = MagicMock()
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = mock_doc
        mock_async_session_local.execute.return_value = mock_res
        
        await process_document_background(doc_id, b"dummy", "dummy.pdf", "path/to/dummy.pdf")
        
        # Assert rollback was called
        mock_async_session_local.rollback.assert_called_once()
        
        # The document status should be set to "failed" (using err_db, which is another instance of AsyncSessionLocal)
        # So we check if mock_doc.status was set to failed (if it was returned)
        assert mock_doc.status == "failed"
