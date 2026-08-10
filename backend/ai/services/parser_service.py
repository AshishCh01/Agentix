import io
from typing import Any, Dict, List
from pypdf import PdfReader
from docx import Document
import pandas as pd
from ai.services.llm_service import llm_service


async def parse_document(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts content from uploaded file broken down by page number.
    Includes a Vision LLM fallback for scanned/image-based PDF pages.
    Returns a list of dicts: [{"page_number": int, "text": str}]
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext == "pdf":
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        pages = []

        for idx, page in enumerate(pdf_reader.pages):
            page_num = idx + 1
            page_text = page.extract_text() or ""
            page_text_clean = page_text.strip()

            # If text extraction returned near-empty content (< 20 chars), attempt Vision fallback for scanned pages
            if len(page_text_clean) < 20:
                rendered_img_bytes = None

                # Try rendering page via PyMuPDF (fitz) if installed
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    fitz_page = doc[idx]
                    pix = fitz_page.get_pixmap()
                    rendered_img_bytes = pix.tobytes("png")
                    doc.close()
                except ImportError:
                    # Fallback: Extract embedded image directly from pypdf if available
                    if hasattr(page, "images") and len(page.images) > 0:
                        rendered_img_bytes = page.images[0].data

                if rendered_img_bytes:
                    try:
                        ocr_description = await llm_service.describe_image(
                            file_bytes=rendered_img_bytes,
                            mime_type="image/png",
                            prompt=(
                                "Transcribe all visible text, numbers, headings, and data "
                                "tables from this scanned document page accurately."
                            ),
                        )
                        if ocr_description:
                            page_text_clean = ocr_description
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"OCR Extraction failed for page {page_num} of {filename}: {str(e)}")
                        page_text_clean = "[OCR Extraction Failed]"

            pages.append({"page_number": page_num, "text": page_text_clean})

        return pages if pages else [{"page_number": 1, "text": ""}]

    elif ext in ["docx", "doc"]:
        doc = Document(io.BytesIO(file_bytes))
        full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return [{"page_number": 1, "text": full_text}]

    elif ext in ["csv", "xlsx", "xls"]:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return [{"page_number": 1, "text": df.to_string(index=False)}]

    elif ext in ["txt", "md", "json"]:
        full_text = file_bytes.decode("utf-8", errors="ignore")
        return [{"page_number": 1, "text": full_text}]

    elif ext in ["png", "jpg", "jpeg", "webp"]:
        mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        description = await llm_service.describe_image(file_bytes=file_bytes, mime_type=mime_type)
        return [{"page_number": 1, "text": description}]

    else:
        # Fallback for unrecognized extensions (try as utf-8 text)
        try:
            full_text = file_bytes.decode("utf-8")
            return [{"page_number": 1, "text": full_text}]
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported binary file format: .{ext}")