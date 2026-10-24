import io
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pypdf import PdfWriter

from ai.services.parser_service import parse_document
from app.models.document import Document
from app.models.document_chunks import DocumentChunk
from app.routes.upload import process_document_background


def create_blank_pdf(num_pages: int = 1) -> bytes:
    """Create a minimal in-memory PDF with blank pages (extract_text returns empty string)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PHASE 4 TEST SUITE: Issue #22 - Failed OCR Text Must Not Be Indexed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forced_ocr_failure_parser_skips_content():
    """
    Test Case A: Forced OCR failure
    - Mocks the vision LLM provider to raise an exception.
    - Verifies that the affected page's text is empty (not '[OCR Extraction Failed]').
    - Verifies that ocr_failed is flagged as True.
    - Verifies '[OCR Extraction Failed]' does not appear in any returned dictionary.
    """
    pdf_bytes = create_blank_pdf(num_pages=1)
    doc_id = uuid.uuid4()

    with patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr, \
         patch("ai.services.parser_service.logger.warning") as mock_log:
        mock_ocr.side_effect = RuntimeError("Vision service unavailable: 503 Service Unavailable")

        pages = await parse_document(pdf_bytes, "scanned_doc.pdf", doc_id=doc_id)

        assert len(pages) == 1
        page = pages[0]
        assert page["page_number"] == 1
        assert page["text"] == ""
        assert page["ocr_failed"] is True
        assert "[OCR Extraction Failed]" not in page["text"]
        assert "[OCR Extraction Failed]" not in str(pages)

        # Verify contextual warning was logged with doc_id, filename, and page number
        mock_log.assert_called()
        log_messages = [call.args[0] for call in mock_log.call_args_list]
        ocr_warning = next((msg for msg in log_messages if "OCR extraction failed" in msg), None)
        assert ocr_warning is not None
        assert str(doc_id) in ocr_warning
        assert "scanned_doc.pdf" in ocr_warning
        assert "page 1" in ocr_warning


@pytest.mark.asyncio
async def test_successful_ocr_parser_preserves_content():
    """
    Test Case D: Successful OCR
    - Mocks the vision LLM provider to return transcribed text.
    - Verifies that transcribed text is preserved and ocr_failed is False.
    """
    pdf_bytes = create_blank_pdf(num_pages=1)
    doc_id = uuid.uuid4()
    transcribed_text = "Patient Name: John Doe\nDiagnosis: Healthy\nDate: 2026-10-24"

    with patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = transcribed_text

        pages = await parse_document(pdf_bytes, "receipt.pdf", doc_id=doc_id)

        assert len(pages) == 1
        page = pages[0]
        assert page["page_number"] == 1
        assert page["text"] == transcribed_text
        assert page["ocr_failed"] is False


@pytest.mark.asyncio
async def test_partial_ocr_failure_parser():
    """
    Test Case B (Parser level): Partial OCR failure across multi-page document
    - Page 1 succeeds OCR.
    - Page 2 fails OCR with an exception.
    - Verifies page 1 text is preserved, page 2 is skipped with ocr_failed=True.
    """
    pdf_bytes = create_blank_pdf(num_pages=2)
    doc_id = uuid.uuid4()

    with patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.side_effect = [
            "Content from successfully scanned page 1 with details.",
            RuntimeError("OCR timeout on page 2"),
        ]

        pages = await parse_document(pdf_bytes, "multipage.pdf", doc_id=doc_id)

        assert len(pages) == 2
        # Page 1
        assert pages[0]["page_number"] == 1
        assert "successfully scanned page 1" in pages[0]["text"]
        assert pages[0]["ocr_failed"] is False

        # Page 2
        assert pages[1]["page_number"] == 2
        assert pages[1]["text"] == ""
        assert pages[1]["ocr_failed"] is True
        assert "[OCR Extraction Failed]" not in pages[1]["text"]


@pytest.mark.asyncio
async def test_ingestion_partial_ocr_failure():
    """
    Test Case B (Ingestion level): Partial OCR failure in process_document_background
    - Page 1 succeeds OCR and has text.
    - Page 2 fails OCR.
    - Verifies:
      1. Chunks are generated ONLY for page 1.
      2. No chunk or embedding contains page 2 or any error text.
      3. Document status is 'ready'.
      4. Document ocr_warning_count is 1.
      5. Document error_message indicates the partial warning.
    """
    doc_id = uuid.uuid4()
    pdf_bytes = create_blank_pdf(num_pages=2)

    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.status = "processing"
    mock_doc.ocr_warning_count = 0
    mock_doc.error_message = None

    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt).lower()
        if "count(" in stmt_str:
            mock_res.scalar.return_value = 1  # 1 chunk created
        else:
            mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute.side_effect = mock_execute

    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.embedding_service") as mock_es, \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_ocr.side_effect = [
            "Valid page 1 invoice text with lots of important information.",
            RuntimeError("Rate limit exceeded on page 2"),
        ]
        mock_es.generate_batch_embeddings = AsyncMock(return_value=[[0.1] * 1536])

        await process_document_background(doc_id, pdf_bytes, "invoice.pdf", "path/invoice.pdf")

        # Verify only 1 chunk was added (from page 1)
        add_calls = mock_session.add.call_args_list
        assert len(add_calls) == 1
        chunk: DocumentChunk = add_calls[0][0][0]
        assert chunk.page_number == 1
        assert "Valid page 1 invoice text" in chunk.content
        assert "[OCR Extraction Failed]" not in chunk.content

        # Verify embedding service input did not contain error messages
        emb_call_texts = mock_es.generate_batch_embeddings.call_args[0][0]
        assert len(emb_call_texts) == 1
        assert "[OCR Extraction Failed]" not in emb_call_texts[0]

        # Verify document status and warning tracking
        assert mock_doc.status == "ready"
        assert mock_doc.ocr_warning_count == 1
        assert "Warning: OCR extraction failed on 1 page(s)." in mock_doc.error_message
        mock_session.commit.assert_called()


@pytest.mark.asyncio
async def test_ingestion_complete_ocr_failure():
    """
    Test Case C: Complete OCR failure
    - All pages fail OCR.
    - Verifies:
      1. Zero chunks added to DB.
      2. Embedding service is NEVER called.
      3. Document status is marked 'failed'.
      4. Document ocr_warning_count is recorded (e.g. 2).
      5. Informative failure reason is set in error_message.
    """
    doc_id = uuid.uuid4()
    pdf_bytes = create_blank_pdf(num_pages=2)

    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.status = "processing"
    mock_doc.ocr_warning_count = 0
    mock_doc.error_message = None

    async def mock_execute(stmt):
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute.side_effect = mock_execute

    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.embedding_service") as mock_es, \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_ocr.side_effect = RuntimeError("OCR vision API model failure")

        await process_document_background(doc_id, pdf_bytes, "failed_scan.pdf", "path/failed_scan.pdf")

        # Verify NO chunks were added
        mock_session.add.assert_not_called()

        # Verify embedding service was NOT invoked
        mock_es.generate_batch_embeddings.assert_not_called()

        # Verify document transitioned to 'failed' with diagnostic message
        assert mock_doc.status == "failed"
        assert mock_doc.ocr_warning_count == 2
        assert "OCR extraction failed on all 2 page(s)" in mock_doc.error_message
        mock_session.commit.assert_called()


@pytest.mark.asyncio
async def test_downstream_safety_ocr_error_strings_never_chunked_or_embedded():
    """
    Test Case E: Downstream safety
    - Ensures that even if an input somehow contained '[OCR Extraction Failed]',
      the chunking logic and embedding step reject it and refuse to store it.
    """
    doc_id = uuid.uuid4()

    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.status = "processing"
    mock_doc.ocr_warning_count = 0
    mock_doc.error_message = None

    async def mock_execute(stmt):
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute.side_effect = mock_execute

    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.parse_document", new_callable=AsyncMock) as mock_parse, \
         patch("app.routes.upload.embedding_service") as mock_es:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        # Simulate parser returning rogue error string
        mock_parse.return_value = [
            {"page_number": 1, "text": "[OCR Extraction Failed]", "ocr_failed": True}
        ]

        await process_document_background(doc_id, b"dummy", "rogue.pdf", "path/rogue.pdf")

        # Must not produce any chunks or embeddings
        mock_session.add.assert_not_called()
        mock_es.generate_batch_embeddings.assert_not_called()
        assert mock_doc.status == "failed"


@pytest.mark.asyncio
async def test_regression_standard_text_pdf_unaffected():
    """
    Test Case F: Regression coverage for standard text-based PDF
    - When text extraction yields > 20 characters, OCR fallback is bypassed entirely.
    - OCR provider is not called, ocr_failed is False, and text is preserved.
    """
    # Create a mock PDF page that returns extractable text > 20 chars
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a legitimate text-based PDF document with plenty of content."
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page]

    with patch("ai.services.parser_service.PdfReader", return_value=mock_pdf_reader), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        pages = await parse_document(b"%PDF-mock", "standard_text.pdf")

        assert len(pages) == 1
        assert pages[0]["ocr_failed"] is False
        assert "This is a legitimate text-based PDF" in pages[0]["text"]
        # OCR describe_image should not be invoked for standard text
        mock_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_regression_non_pdf_unaffected():
    """
    Test Case F: Regression coverage for non-PDF files (.txt, .md, .json)
    - Verifies that text files continue to parse without OCR or warning counts.
    """
    text_content = "Plain text document content for testing regression."
    pages = await parse_document(text_content.encode("utf-8"), "test.txt")
    assert len(pages) == 1
    assert pages[0]["text"] == text_content
    assert pages[0].get("ocr_failed", False) is False
