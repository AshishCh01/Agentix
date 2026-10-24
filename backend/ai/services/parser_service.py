import codecs
import io
import logging
from typing import Any, Dict, List
import charset_normalizer
from pypdf import PdfReader
from docx import Document
import pandas as pd
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

BOM_UTF32_LE = codecs.BOM_UTF32_LE
BOM_UTF32_BE = codecs.BOM_UTF32_BE
BOM_UTF8 = codecs.BOM_UTF8
BOM_UTF16_LE = codecs.BOM_UTF16_LE
BOM_UTF16_BE = codecs.BOM_UTF16_BE

WESTERN_ENCODINGS = {"cp1252", "windows_1252", "latin_1", "iso8859_1", "iso_8859_1"}


def decode_text_bytes(file_bytes: bytes, filename: str = "") -> str:
    """
    Robustly decode raw text bytes into a string.

    1. Returns empty string for empty inputs.
    2. Detects and handles BOMs in order: UTF-32 (4-byte), UTF-8 (3-byte), UTF-16 (2-byte).
    3. Detects UTF-16 without BOM via alternating null-byte distribution.
    4. Attempts strict UTF-8 decoding for non-BOM inputs.
    5. For non-UTF-8 inputs, uses charset-normalizer to detect plausible encodings.
    6. Falls back to UTF-8 with errors='replace' if detection fails or is unreliable,
       logging an appropriate warning. Never silently discards bytes with errors='ignore'.
    """
    if not file_bytes:
        return ""

    # 1. Check for Byte Order Marks (BOM) in descending order of prefix length
    # Check 4-byte UTF-32 BOMs first so UTF-32-LE is not confused with UTF-16-LE
    if file_bytes.startswith(BOM_UTF32_LE) or file_bytes.startswith(BOM_UTF32_BE):
        try:
            return file_bytes.decode("utf-32")
        except UnicodeDecodeError as exc:
            logger.warning(
                f"Failed strict UTF-32 decoding for '{filename or 'document'}' with BOM: {exc}. "
                "Falling back to UTF-32 with errors='replace'."
            )
            return file_bytes.decode("utf-32", errors="replace")

    # Check 3-byte UTF-8 BOM
    if file_bytes.startswith(BOM_UTF8):
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            logger.warning(
                f"Failed strict UTF-8 decoding for '{filename or 'document'}' with UTF-8 BOM: {exc}. "
                "Falling back to UTF-8 with errors='replace'."
            )
            return file_bytes.decode("utf-8-sig", errors="replace")

    # Check 2-byte UTF-16 BOMs
    if file_bytes.startswith(BOM_UTF16_LE) or file_bytes.startswith(BOM_UTF16_BE):
        try:
            return file_bytes.decode("utf-16")
        except UnicodeDecodeError as exc:
            logger.warning(
                f"Failed strict UTF-16 decoding for '{filename or 'document'}' with BOM: {exc}. "
                "Falling back to UTF-16 with errors='replace'."
            )
            return file_bytes.decode("utf-16", errors="replace")

    # 2. Check for UTF-16 without BOM (characterized by alternating null bytes for ASCII code points)
    if len(file_bytes) >= 4 and len(file_bytes) % 2 == 0:
        odd_nulls = file_bytes[1::2].count(0)
        even_nulls = file_bytes[0::2].count(0)
        half_len = len(file_bytes) // 2
        # If at least half of the code points are in ASCII range (very typical of text)
        if odd_nulls >= half_len // 2:
            try:
                return file_bytes.decode("utf-16-le")
            except UnicodeDecodeError:
                pass
        elif even_nulls >= half_len // 2:
            try:
                return file_bytes.decode("utf-16-be")
            except UnicodeDecodeError:
                pass

    # 3. Strict UTF-8 decoding when no BOM is present
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 4. Detect encoding with charset-normalizer
    try:
        matches = charset_normalizer.from_bytes(file_bytes)
        best = matches.best()
    except Exception as exc:
        logger.warning(f"Charset detection failed for '{filename or 'document'}': {exc}")
        best = None

    if best is not None:
        # Avoid false-positive UTF-16/32 detections on 8-bit text without BOM.
        # UTF-16/32 Latin/ASCII text contains null bytes for ASCII code points.
        # If there are no null bytes, UTF-16/32 is an erroneous match.
        has_nulls = b"\x00" in file_bytes
        filtered_matches = [
            m for m in matches
            if has_nulls or not m.encoding.startswith(("utf_16", "utf_32"))
        ]

        if filtered_matches:
            best_candidate = filtered_matches[0]
            chosen_match = None

            # Prioritize standard Western encoding (cp1252 / latin-1) when candidates
            # have low chaos (<= best_candidate.chaos + 0.15 and <= 0.20).
            # This avoids choosing niche Baltic/Central European codepages (cp1257, cp775)
            # that misinterpret Western accents (French/Spanish) or Windows-1252 quotes/euro.
            for m in filtered_matches:
                all_encs = {m.encoding.lower().replace("-", "_")} | {
                    s.encoding.lower().replace("-", "_") for s in getattr(m, "submatch", [])
                }
                if any(enc in all_encs for enc in WESTERN_ENCODINGS):
                    if m.chaos <= best_candidate.chaos + 0.15 and m.chaos <= 0.20:
                        chosen_match = "cp1252"
                        break

            if not chosen_match and best_candidate.chaos <= 0.4:
                chosen_match = best_candidate.encoding

            if chosen_match:
                try:
                    return file_bytes.decode(chosen_match)
                except UnicodeDecodeError as exc:
                    logger.warning(
                        f"Failed strict decoding for '{filename or 'document'}' with detected encoding '{chosen_match}': {exc}. "
                        f"Falling back to {chosen_match} with errors='replace'."
                    )
                    return file_bytes.decode(chosen_match, errors="replace")
                except LookupError as exc:
                    logger.warning(f"Unknown encoding '{chosen_match}' for '{filename or 'document'}': {exc}")

    # 5. Fallback when detection fails or confidence is insufficient
    logger.warning(
        f"Encoding detection failed or insufficient confidence for '{filename or 'document'}'. "
        "Falling back to UTF-8 with errors='replace'."
    )
    return file_bytes.decode("utf-8", errors="replace")


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
                        logger.error(f"OCR Extraction failed for page {page_num} of {filename}: {str(e)}")
                        page_text_clean = "[OCR Extraction Failed]"

            pages.append({"page_number": page_num, "text": page_text_clean})

        return pages if pages else [{"page_number": 1, "text": ""}]

    elif ext == "doc":
        # Legacy binary .doc (OLE2/CFBF) is a different format from OOXML
        # .docx -- python-docx's Document() only reads .docx (a zip
        # container) and raises an opaque zip-parsing error on real .doc
        # files. Fail fast here with a clear, actionable message instead of
        # accepting the upload and letting it die deep inside python-docx.
        raise ValueError(
            "The legacy .doc format is not supported. Please save this file as "
            ".docx (in Word: File > Save As > Word Document) and re-upload."
        )

    elif ext == "docx":
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
        full_text = decode_text_bytes(file_bytes, filename)
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