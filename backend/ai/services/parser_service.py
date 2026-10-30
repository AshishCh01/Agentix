import asyncio
import codecs
import io
import logging
from typing import Any, Dict, List, Optional
import charset_normalizer
from pypdf import PdfReader
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
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


def detect_image_mime_type(image_bytes: bytes) -> Optional[str]:
    """
    Detect the MIME type of image bytes using magic byte signatures with Pillow fallback.
    Returns standard MIME type string (e.g. 'image/jpeg', 'image/png', 'image/webp')
    or None if unrecognized or unsupported.
    """
    if not image_bytes or len(image_bytes) < 4:
        return None

    # Magic byte signatures
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(image_bytes) >= 12 and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"II*\x00") or image_bytes.startswith(b"MM\x00*"):
        return "image/tiff"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return "image/jp2"

    # Pillow fallback if available
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "").upper()
            mapping = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "GIF": "image/gif",
                "TIFF": "image/tiff",
                "BMP": "image/bmp",
                "JP2": "image/jp2",
            }
            return mapping.get(fmt)
    except Exception:
        pass

    return None


def should_trigger_ocr(page_text: str, has_images: bool = False) -> bool:
    """
    Determines whether OCR / Vision fallback should be triggered for a PDF page.

    Triggers OCR if:
    - The page has zero extractable text (potential scanned page).
    - The page has extracted text, but it is corrupted / low-quality
      (printable ratio < 70% or no valid words) and has images.
    - The page has sparse native text (< 50 chars) AND contains images
      (e.g. scanned document with only a header or page number recognized natively).

    Does NOT trigger OCR if:
    - The page has substantial, high-quality native text (>= 50 chars, high printable ratio).
    - The page is legitimately sparse text (short heading, title, page number) with NO images.
    """
    clean = (page_text or "").strip()

    if not clean:
        # Zero native text: scanned document page requires OCR fallback
        return True

    # Assess quality of extracted text
    printable_count = sum(1 for c in clean if c.isprintable())
    printable_ratio = printable_count / len(clean)
    words = [w for w in clean.split() if any(c.isalnum() for c in w)]

    # Corrupted / unmapped font artifacts (e.g. low printable ratio or no alphanumeric words)
    is_corrupted = printable_ratio < 0.70 or (len(clean) >= 5 and len(words) == 0)
    if is_corrupted and has_images:
        return True

    # Substantial, high-quality native text: native text is sufficient, skip OCR
    if len(clean) >= 50 and printable_ratio >= 0.85 and len(words) >= 5:
        return False

    # Sparse native text (< 50 chars):
    # If the page contains images, OCR is needed to capture the image content
    if has_images:
        return True

    # Sparse native text with NO images: legitimately sparse page (e.g. "Chapter 1", "Page 5")
    return False


def format_docx_table(table: Any) -> str:
    """
    Converts a python-docx Table into a readable Markdown table format.
    Handles empty cells, multiple rows/columns, cell newlines, pipes,
    and horizontally merged cells without duplicating cell content.
    """
    if not hasattr(table, "rows") or not table.rows:
        return ""

    rows_data: List[List[str]] = []
    for row in table.rows:
        row_cells: List[str] = []
        seen_tc_ids = set()
        for cell in row.cells:
            # Handle horizontally merged cells: python-docx references the same _tc element
            tc_elem = getattr(cell, "_tc", cell)
            tc_id = id(tc_elem)
            if tc_id in seen_tc_ids:
                row_cells.append("")
            else:
                seen_tc_ids.add(tc_id)
                clean_text = cell.text.strip().replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")
                row_cells.append(clean_text)
        rows_data.append(row_cells)

    # Return empty string if all cells across the table are whitespace
    if not any(any(c for c in r) for r in rows_data):
        return ""

    num_cols = max(len(r) for r in rows_data)
    if num_cols == 0:
        return ""

    normalized_rows = [r + [""] * (num_cols - len(r)) for r in rows_data]

    lines = []
    header_row = normalized_rows[0]
    lines.append("| " + " | ".join(header_row) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    for data_row in normalized_rows[1:]:
        lines.append("| " + " | ".join(data_row) + " |")

    return "\n".join(lines)


def extract_docx_headers_and_footers(doc: Any) -> tuple[List[str], List[str]]:
    """
    Extracts relevant document headers and footers across sections,
    deduplicating identical header/footer content (e.g., from linked sections).
    """
    headers: List[str] = []
    footers: List[str] = []
    seen_headers: set = set()
    seen_footers: set = set()

    if not hasattr(doc, "sections"):
        return headers, footers

    for section in doc.sections:
        # 1. Header extraction
        header_containers = []
        if hasattr(section, "header") and section.header:
            header_containers.append(section.header)
        if getattr(section, "different_first_page_header_footer", False):
            if hasattr(section, "first_page_header") and section.first_page_header:
                header_containers.append(section.first_page_header)

        for container in header_containers:
            for p in getattr(container, "paragraphs", []):
                txt = p.text.strip()
                if txt and txt not in seen_headers:
                    seen_headers.add(txt)
                    headers.append(txt)
            for tbl in getattr(container, "tables", []):
                tbl_txt = format_docx_table(tbl)
                if tbl_txt and tbl_txt not in seen_headers:
                    seen_headers.add(tbl_txt)
                    headers.append(tbl_txt)

        # 2. Footer extraction
        footer_containers = []
        if hasattr(section, "footer") and section.footer:
            footer_containers.append(section.footer)
        if getattr(section, "different_first_page_header_footer", False):
            if hasattr(section, "first_page_footer") and section.first_page_footer:
                footer_containers.append(section.first_page_footer)

        for container in footer_containers:
            for p in getattr(container, "paragraphs", []):
                txt = p.text.strip()
                if txt and txt not in seen_footers:
                    seen_footers.add(txt)
                    footers.append(txt)
            for tbl in getattr(container, "tables", []):
                tbl_txt = format_docx_table(tbl)
                if tbl_txt and tbl_txt not in seen_footers:
                    seen_footers.add(tbl_txt)
                    footers.append(tbl_txt)

    return headers, footers


def iter_docx_body_blocks(doc: Any):
    """
    Yields extracted text blocks from the docx document body,
    preserving the exact sequential order of paragraphs and tables.
    """
    def _traverse(parent_elem):
        for child in parent_elem:
            try:
                tag = getattr(child, "tag", "")
                if tag.endswith("p"):
                    p = Paragraph(child, doc)
                    txt = p.text.strip()
                    if txt:
                        yield txt
                elif tag.endswith("tbl"):
                    tbl = Table(child, doc)
                    tbl_txt = format_docx_table(tbl)
                    if tbl_txt:
                        yield tbl_txt
                elif tag.endswith("sdt"):
                    sdt_content = child.find(qn("w:sdtContent"))
                    container = sdt_content if sdt_content is not None else child
                    yield from _traverse(container)
            except Exception as block_err:
                logger.debug(f"Skipping malformed block in docx body: {block_err}")

    try:
        if hasattr(doc, "_element") and hasattr(doc._element, "body"):
            yield from _traverse(doc._element.body)
        else:
            raise AttributeError("Document has no _element.body")
    except Exception as iter_err:
        logger.warning(
            f"Error during ordered docx body iteration: {iter_err}. Falling back to paragraphs and tables."
        )
        for p in getattr(doc, "paragraphs", []):
            txt = p.text.strip()
            if txt:
                yield txt
        for tbl in getattr(doc, "tables", []):
            tbl_txt = format_docx_table(tbl)
            if tbl_txt:
                yield tbl_txt


def format_excel_sheet(sheet_name: str, df: Any) -> str:
    """
    Format a single worksheet DataFrame into readable text with a sheet header.
    Handles empty sheets, missing values (NaN), and preserves rows, columns, and data.
    """
    if df is None:
        return f"Sheet: {sheet_name}"

    if df.empty:
        if len(df.columns) > 0 and any(not str(c).startswith("Unnamed:") for c in df.columns):
            cols_str = "  ".join(str(c) for c in df.columns if not str(c).startswith("Unnamed:"))
            return f"Sheet: {sheet_name}\n{cols_str}".strip()
        return f"Sheet: {sheet_name}"

    table_str = df.to_string(index=False).strip()
    if not table_str or table_str.startswith("Empty DataFrame"):
        return f"Sheet: {sheet_name}"

    return f"Sheet: {sheet_name}\n{table_str}"


async def parse_document(
    file_bytes: bytes,
    filename: str,
    doc_id: Any = None,
) -> List[Dict[str, Any]]:
    """
    Extracts content from uploaded file broken down by page number.
    Includes a Vision LLM fallback for scanned/image-based PDF pages.
    Returns a list of dicts: [{"page_number": int, "text": str, "ocr_failed": bool}]
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext == "pdf":
        fitz_doc = None
        pages = []

        try:
            # 1. Open PDF once with PyMuPDF (fitz) if available to avoid per-page reopening overhead
            try:
                import fitz
                fitz_doc = fitz.open(stream=file_bytes, filetype="pdf")
            except ImportError:
                fitz_doc = None
            except Exception as open_err:
                logger.warning(
                    f"Could not open PDF with PyMuPDF for {filename}: {type(open_err).__name__} - {str(open_err)[:100]}"
                )
                fitz_doc = None

            pdf_reader = PdfReader(io.BytesIO(file_bytes))

            for idx, page in enumerate(pdf_reader.pages):
                page_num = idx + 1
                page_text = page.extract_text() or ""
                page_text_clean = page_text.strip()
                ocr_failed = False

                # Check if page has images
                has_images = False
                fitz_page = None
                if fitz_doc is not None:
                    try:
                        fitz_page = fitz_doc[idx]
                        has_images = len(fitz_page.get_images()) > 0
                    except Exception:
                        fitz_page = None

                if not has_images and hasattr(page, "images"):
                    try:
                        has_images = len(page.images) > 0
                    except Exception:
                        pass

                # Check if OCR is needed using refined decision logic
                if should_trigger_ocr(page_text_clean, has_images=has_images):
                    rendered_img_bytes = None
                    detected_mime_type = None

                    # A. Try rendering page at 2x scale (144 DPI) via fitz
                    if fitz_page is not None:
                        try:
                            import fitz as fitz_module
                            # 2x scale: 72 DPI standard coordinate space * 2 = 144 DPI
                            matrix = fitz_module.Matrix(2.0, 2.0)
                            pix = fitz_page.get_pixmap(matrix=matrix)
                            rendered_img_bytes = pix.tobytes("png")
                            pix = None  # Free pixmap buffer immediately to conserve memory
                            detected_mime_type = "image/png"
                        except Exception as render_err:
                            logger.warning(
                                f"Failed to render page {page_num} of {filename} at 2x scale: {type(render_err).__name__}"
                            )

                    # B. Fallback: extract embedded image directly from pypdf if rendering unavailable
                    if not rendered_img_bytes and hasattr(page, "images") and len(page.images) > 0:
                        try:
                            img_data = page.images[0].data
                            img_mime = detect_image_mime_type(img_data)
                            if img_mime:
                                rendered_img_bytes = img_data
                                detected_mime_type = img_mime
                            else:
                                logger.warning(
                                    f"Embedded image on page {page_num} of {filename} has unsupported format"
                                )
                                ocr_failed = True
                        except Exception as img_err:
                            logger.warning(
                                f"Failed to extract embedded image on page {page_num} of {filename}: {type(img_err).__name__}"
                            )
                            ocr_failed = True

                    # C. Call OCR/Vision model if valid image bytes and MIME type are available
                    if rendered_img_bytes and detected_mime_type:
                        try:
                            ocr_description = await llm_service.describe_image(
                                file_bytes=rendered_img_bytes,
                                mime_type=detected_mime_type,
                                prompt=(
                                    "Transcribe all visible text, numbers, headings, and data "
                                    "tables from this scanned document page accurately."
                                ),
                            )
                            if ocr_description:
                                ocr_clean = ocr_description.strip()
                                # Preserve native text if present and combine without duplication
                                if not page_text_clean:
                                    page_text_clean = ocr_clean
                                elif page_text_clean.lower() in ocr_clean.lower():
                                    page_text_clean = ocr_clean
                                elif ocr_clean.lower() in page_text_clean.lower():
                                    pass
                                else:
                                    page_text_clean = f"{page_text_clean}\n\n{ocr_clean}"
                        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as e:
                            doc_context = f"document {doc_id} ({filename})" if doc_id else f"{filename}"
                            logger.warning(
                                f"⚠️ OCR extraction failed for {doc_context} on page {page_num}: {type(e).__name__} - {str(e)[:150]}"
                            )
                            # Safe OCR failure handling from Issue #22
                            page_text_clean = ""
                            ocr_failed = True

                pages.append({
                    "page_number": page_num,
                    "text": page_text_clean,
                    "ocr_failed": ocr_failed,
                })

        finally:
            # Safe resource management: ensure PDF document is always closed
            if fitz_doc is not None:
                try:
                    fitz_doc.close()
                except Exception:
                    pass

        return pages if pages else [{"page_number": 1, "text": "", "ocr_failed": False}]

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
        if not file_bytes:
            return [{"page_number": 1, "text": ""}]

        doc = Document(io.BytesIO(file_bytes))
        headers, footers = extract_docx_headers_and_footers(doc)
        body_blocks = list(iter_docx_body_blocks(doc))

        all_parts = headers + body_blocks + footers
        full_text = "\n\n".join([p for p in all_parts if p.strip()])
        return [{"page_number": 1, "text": full_text}]

    elif ext == "csv":
        if not file_bytes:
            return [{"page_number": 1, "text": ""}]
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            if df.empty:
                return [{"page_number": 1, "text": ""}]
            return [{"page_number": 1, "text": df.to_string(index=False)}]
        except Exception as e:
            logger.warning(f"Failed to parse CSV file {filename}: {e}")
            return [{"page_number": 1, "text": ""}]

    elif ext in ["xlsx", "xls"]:
        if not file_bytes:
            return [{"page_number": 1, "text": ""}]

        try:
            sheets_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        except Exception as e:
            logger.warning(f"Failed to read Excel file {filename}: {e}")
            return [{"page_number": 1, "text": ""}]

        if not sheets_dict:
            return [{"page_number": 1, "text": ""}]

        pages = []
        for idx, (sheet_name, df) in enumerate(sheets_dict.items()):
            sheet_text = format_excel_sheet(str(sheet_name), df)
            pages.append({
                "page_number": idx + 1,
                "text": sheet_text,
            })

        return pages if pages else [{"page_number": 1, "text": ""}]

    elif ext in ["txt", "md", "json"]:
        full_text = decode_text_bytes(file_bytes, filename)
        return [{"page_number": 1, "text": full_text}]

    elif ext in ["png", "jpg", "jpeg", "webp"]:
        detected_mime = detect_image_mime_type(file_bytes)
        fallback_mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        mime_type = detected_mime or fallback_mime
        description = await llm_service.describe_image(file_bytes=file_bytes, mime_type=mime_type)
        return [{"page_number": 1, "text": description}]

    else:
        # Fallback for unrecognized extensions (try as utf-8 text)
        try:
            full_text = file_bytes.decode("utf-8")
            return [{"page_number": 1, "text": full_text}]
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported binary file format: .{ext}")