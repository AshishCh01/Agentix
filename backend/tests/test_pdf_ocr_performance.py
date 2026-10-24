import io
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pypdf import PdfWriter

from ai.services.parser_service import (
    detect_image_mime_type,
    parse_document,
    should_trigger_ocr,
)
from app.models.document import Document
from app.routes.upload import process_document_background


def create_blank_pdf(num_pages: int = 1) -> bytes:
    """Create a minimal in-memory PDF with blank pages (extract_text returns empty string)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ===========================================================================
# GROUP A: PDF Rendering and Resource Management
# ===========================================================================

@pytest.mark.asyncio
async def test_pdf_rendered_at_2x_scale():
    """
    Verify that PDF pages rendered for OCR use 2x scale (144 DPI equivalent)
    via fitz.Matrix(2.0, 2.0).
    """
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\nrendered"
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    pdf_bytes = create_blank_pdf(num_pages=1)

    with patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Transcribed high-res page text"

        pages = await parse_document(pdf_bytes, "high_res_scan.pdf")

        assert len(pages) == 1
        assert pages[0]["text"] == "Transcribed high-res page text"

        # Verify get_pixmap was called with a 2x scale matrix
        mock_fitz.Matrix.assert_called_with(2.0, 2.0)
        mock_page.get_pixmap.assert_called_once_with(matrix=mock_fitz.Matrix.return_value)


@pytest.mark.asyncio
async def test_pdf_opened_only_once_for_multipage_document():
    """
    Verify that the PDF document is opened only once per parse_document call,
    reusing the document object across multiple pages rather than reopening per page.
    """
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\nmultipage"
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    pdf_bytes = create_blank_pdf(num_pages=4)

    with patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Page content"

        pages = await parse_document(pdf_bytes, "multipage_scan.pdf")

        assert len(pages) == 4
        # CRITICAL CHECK: fitz.open must be called exactly ONCE, not 4 times
        assert mock_fitz.open.call_count == 1


@pytest.mark.asyncio
async def test_pdf_document_closed_on_success_and_failure():
    """
    Verify safe resource management: fitz_doc.close() is called both on
    successful parse and when an exception occurs.
    """
    # 1. On success
    mock_fitz_succ = MagicMock()
    mock_doc_succ = MagicMock()
    mock_page_succ = MagicMock()
    mock_pix_succ = MagicMock()
    mock_pix_succ.tobytes.return_value = b"\x89PNG\r\n\x1a\nsuccess"
    mock_page_succ.get_pixmap.return_value = mock_pix_succ
    mock_doc_succ.__getitem__.return_value = mock_page_succ
    mock_fitz_succ.open.return_value = mock_doc_succ

    pdf_bytes = create_blank_pdf(num_pages=2)

    with patch.dict("sys.modules", {"fitz": mock_fitz_succ}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Success content"
        await parse_document(pdf_bytes, "success.pdf")
        assert mock_doc_succ.close.called

    # 2. On failure during iteration
    mock_fitz_fail = MagicMock()
    mock_doc_fail = MagicMock()
    mock_fitz_fail.open.return_value = mock_doc_fail

    with patch.dict("sys.modules", {"fitz": mock_fitz_fail}), \
         patch("ai.services.parser_service.PdfReader", side_effect=RuntimeError("Corrupt PDF structure")):
        with pytest.raises(RuntimeError):
            await parse_document(pdf_bytes, "fail.pdf")
        # Resource must still be closed in finally block
        assert mock_doc_fail.close.called


@pytest.mark.asyncio
async def test_page_order_and_numbers_preserved():
    """Verify that multi-page documents strictly preserve page order and numbering."""
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\npages"

    page_mocks = []
    for i in range(3):
        p = MagicMock()
        p.get_pixmap.return_value = mock_pix
        page_mocks.append(p)
    mock_doc.__getitem__.side_effect = lambda idx: page_mocks[idx]
    mock_fitz.open.return_value = mock_doc

    pdf_bytes = create_blank_pdf(num_pages=3)

    with patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.side_effect = [
            "Content for Page 1",
            "Content for Page 2",
            "Content for Page 3",
        ]

        pages = await parse_document(pdf_bytes, "ordered_doc.pdf")

        assert len(pages) == 3
        assert pages[0] == {"page_number": 1, "text": "Content for Page 1", "ocr_failed": False}
        assert pages[1] == {"page_number": 2, "text": "Content for Page 2", "ocr_failed": False}
        assert pages[2] == {"page_number": 3, "text": "Content for Page 3", "ocr_failed": False}


# ===========================================================================
# GROUP B: OCR Decision Logic
# ===========================================================================

def test_should_trigger_ocr_decision_matrix():
    """Unit test for should_trigger_ocr decision matrix."""
    # 1. Normal substantial text -> No OCR
    substantial_text = "This is a normal paragraph with plenty of clear printable words that easily exceed the threshold."
    assert should_trigger_ocr(substantial_text, has_images=False) is False
    assert should_trigger_ocr(substantial_text, has_images=True) is False

    # 2. Empty text (scanned page) -> OCR needed
    assert should_trigger_ocr("", has_images=False) is True
    assert should_trigger_ocr("", has_images=True) is True
    assert should_trigger_ocr("   \n\t  ", has_images=True) is True

    # 3. Legitimately sparse native text (< 20 or < 50 chars) WITHOUT images -> No OCR!
    assert should_trigger_ocr("Chapter 1", has_images=False) is False
    assert should_trigger_ocr("Table of Contents", has_images=False) is False
    assert should_trigger_ocr("Page 42", has_images=False) is False
    assert should_trigger_ocr("Appendix A: Reference Data", has_images=False) is False

    # 4. Sparse native text WITH images (e.g. scanned receipt with header) -> OCR needed!
    assert should_trigger_ocr("Invoice #101", has_images=True) is True
    assert should_trigger_ocr("Page 1", has_images=True) is True

    # 5. Corrupted / unmapped font artifacts WITH images -> OCR needed!
    corrupted_text = "\x00\x01\x02\x03\x04\x05\x06\x07"
    assert should_trigger_ocr(corrupted_text, has_images=True) is True


@pytest.mark.asyncio
async def test_legitimately_sparse_text_page_not_sent_to_ocr_parser():
    """
    Verify that a legitimately sparse text page (such as 'Chapter 1' or 'Table of Contents')
    with no embedded images does NOT trigger OCR.
    """
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Chapter 1"
    mock_page.images = []

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page]

    with patch("ai.services.parser_service.PdfReader", return_value=mock_pdf_reader), \
         patch.dict("sys.modules", {"fitz": None}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        pages = await parse_document(b"%PDF-mock", "book.pdf")

        assert len(pages) == 1
        assert pages[0]["text"] == "Chapter 1"
        assert pages[0]["ocr_failed"] is False
        # OCR must NOT have been called for this legitimately sparse page!
        mock_ocr.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_document_handling():
    """
    Verify mixed document handling:
    - Page 1: normal text -> no OCR.
    - Page 2: legitimately sparse ("Section 2") without images -> no OCR.
    - Page 3: scanned page (0 text) -> OCR called and preserved.
    """
    p1 = MagicMock()
    p1.extract_text.return_value = "This is a full page of clear text that has plenty of length and details to index."
    p1.images = []

    p2 = MagicMock()
    p2.extract_text.return_value = "Section 2: Diagram"
    p2.images = []

    p3 = MagicMock()
    p3.extract_text.return_value = ""
    p3.images = []

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [p1, p2, p3]

    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\npage3"

    mock_fitz_p1 = MagicMock()
    mock_fitz_p1.get_images.return_value = []

    mock_fitz_p2 = MagicMock()
    mock_fitz_p2.get_images.return_value = []

    mock_fitz_p3 = MagicMock()
    mock_fitz_p3.get_images.return_value = [("xref", 0, 100, 100)]
    mock_fitz_p3.get_pixmap.return_value = mock_pix

    mock_doc.__getitem__.side_effect = lambda idx: [mock_fitz_p1, mock_fitz_p2, mock_fitz_p3][idx]
    mock_fitz.open.return_value = mock_doc

    with patch("ai.services.parser_service.PdfReader", return_value=mock_pdf_reader), \
         patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Extracted text from page 3 scanned diagram."

        pages = await parse_document(b"%PDF-mock", "mixed.pdf")

        assert len(pages) == 3
        # Page 1: native text
        assert "full page of clear text" in pages[0]["text"]
        # Page 2: sparse text without OCR
        assert pages[1]["text"] == "Section 2: Diagram"
        # Page 3: OCR transcribed text
        assert pages[2]["text"] == "Extracted text from page 3 scanned diagram."
        # describe_image should be called only for page 3
        assert mock_ocr.call_count == 1


@pytest.mark.asyncio
async def test_native_and_ocr_text_merge_avoiding_duplicates():
    """
    Verify that when a page has both native text and OCR text:
    - If OCR output already subsumes native text, duplicate text is avoided.
    - If both have distinct information, they are merged cleanly.
    """
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Invoice #12345"
    mock_page.images = [MagicMock()]

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page]

    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_fitz_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\ninvoice"
    mock_fitz_page.get_images.return_value = [("xref",)]
    mock_fitz_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_fitz_page
    mock_fitz.open.return_value = mock_doc

    with patch("ai.services.parser_service.PdfReader", return_value=mock_pdf_reader), \
         patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        # Case 1: OCR contains the native text
        mock_ocr.return_value = "Invoice #12345\nTotal: $1,250.00\nDate: 2026-10-24"
        pages = await parse_document(b"%PDF-mock", "inv.pdf")
        assert pages[0]["text"] == "Invoice #12345\nTotal: $1,250.00\nDate: 2026-10-24"
        assert pages[0]["text"].count("Invoice #12345") == 1

        # Case 2: Distinct content is combined
        mock_ocr.return_value = "Vendor: Acme Corp\nTotal Due: $500"
        pages2 = await parse_document(b"%PDF-mock", "inv.pdf")
        assert pages2[0]["text"] == "Invoice #12345\n\nVendor: Acme Corp\nTotal Due: $500"


# ===========================================================================
# GROUP C: MIME-Type Detection
# ===========================================================================

def test_detect_image_mime_type_all_formats():
    """Verify MIME-type detection for all supported formats and invalid inputs."""
    # PNG
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert detect_image_mime_type(png_bytes) == "image/png"

    # JPEG
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
    assert detect_image_mime_type(jpeg_bytes) == "image/jpeg"

    # WebP
    webp_bytes = b"RIFF\x1a\x00\x00\x00WEBPVP8 "
    assert detect_image_mime_type(webp_bytes) == "image/webp"

    # GIF
    assert detect_image_mime_type(b"GIF89a\x01\x00\x01\x00") == "image/gif"
    assert detect_image_mime_type(b"GIF87a\x01\x00\x01\x00") == "image/gif"

    # TIFF
    assert detect_image_mime_type(b"II*\x00\x08\x00\x00\x00") == "image/tiff"
    assert detect_image_mime_type(b"MM\x00*\x00\x00\x00\x08") == "image/tiff"

    # BMP
    assert detect_image_mime_type(b"BM\x36\x00\x00\x00\x00") == "image/bmp"

    # JP2
    assert detect_image_mime_type(b"\x00\x00\x00\x0cjP  \r\n\x87\n") == "image/jp2"

    # Invalid / unsupported
    assert detect_image_mime_type(b"") is None
    assert detect_image_mime_type(b"short") is None
    assert detect_image_mime_type(b"This is just plain text content.") is None
    assert detect_image_mime_type(b"\x00\x01\x02\x03\x04\x05") is None


@pytest.mark.asyncio
async def test_rendered_page_passed_as_image_png():
    """Verify that PyMuPDF rendered pages are passed with mime_type='image/png'."""
    mock_fitz = MagicMock()
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\nrendered_png"
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    pdf_bytes = create_blank_pdf(num_pages=1)

    with patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Rendered OCR"

        await parse_document(pdf_bytes, "doc.pdf")

        mock_ocr.assert_called_once()
        call_kwargs = mock_ocr.call_args[1]
        assert call_kwargs["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_pypdf_embedded_image_jpeg_passed_as_image_jpeg():
    """
    Verify that an embedded JPEG image extracted via pypdf is passed to describe_image
    with mime_type='image/jpeg', not hardcoded 'image/png'.
    """
    mock_img = MagicMock()
    # Real JPEG magic bytes
    mock_img.data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_page.images = [mock_img]

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page]

    with patch.dict("sys.modules", {"fitz": None}), \
         patch("ai.services.parser_service.PdfReader", return_value=mock_pdf_reader), \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Embedded JPEG Transcribed"

        pages = await parse_document(b"%PDF-mock", "scanned_jpeg.pdf")

        assert len(pages) == 1
        assert pages[0]["text"] == "Embedded JPEG Transcribed"
        mock_ocr.assert_called_once()
        # CRITICAL CHECK: must be image/jpeg, NOT hardcoded image/png!
        assert mock_ocr.call_args[1]["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_direct_image_upload_detects_real_mime():
    """
    Verify that direct image file uploads inspect the actual bytes
    to detect MIME type even if filename extension differs.
    """
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01"

    with patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:
        mock_ocr.return_value = "Receipt description"

        # File named .png but contains JPEG bytes
        await parse_document(jpeg_bytes, "receipt.png")

        mock_ocr.assert_called_once()
        assert mock_ocr.call_args[1]["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_scanned_pdf_end_to_end_ingestion():
    """
    Integration test verifying that a scanned multi-page PDF progresses
    through parsing, 2x scale rendering, OCR transcription, and chunking
    into ready status with clean indexed content.
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
            mock_res.scalar.return_value = 2  # 2 chunks created
        else:
            mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute.side_effect = mock_execute

    mock_fitz = MagicMock()
    mock_doc_fitz = MagicMock()
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\nscanned_bytes"
    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc_fitz.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc_fitz

    with patch.dict("sys.modules", {"fitz": mock_fitz}), \
         patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.embedding_service") as mock_es, \
         patch("ai.services.parser_service.llm_service.describe_image", new_callable=AsyncMock) as mock_ocr:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_ocr.side_effect = [
            "Scanned page 1: Quarterly Sales Revenue reached $1.2M with strong performance.",
            "Scanned page 2: Operational Expenses breakdown indicates high efficiency in Q3.",
        ]
        mock_es.generate_batch_embeddings = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

        await process_document_background(doc_id, pdf_bytes, "quarterly_scan.pdf", "path/quarterly_scan.pdf")

        # Verify 2 chunks were added with readable transcribed text
        add_calls = mock_session.add.call_args_list
        assert len(add_calls) == 2
        chunk1 = add_calls[0][0][0]
        chunk2 = add_calls[1][0][0]
        assert "Quarterly Sales Revenue reached $1.2M" in chunk1.content
        assert "Operational Expenses breakdown" in chunk2.content

        # Verify document status is ready and no OCR errors occurred
        assert mock_doc.status == "ready"
        assert mock_doc.ocr_warning_count == 0
        assert mock_doc.error_message is None
        mock_session.commit.assert_called()
