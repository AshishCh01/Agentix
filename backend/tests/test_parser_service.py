import codecs
import logging
import pytest
from ai.services.parser_service import decode_text_bytes, parse_document


@pytest.mark.asyncio
async def test_decode_empty_file():
    assert decode_text_bytes(b"", "empty.txt") == ""
    pages = await parse_document(b"", "empty.txt")
    assert pages == [{"page_number": 1, "text": ""}]


@pytest.mark.asyncio
async def test_decode_plain_ascii():
    sample = "The quick brown fox jumps over the lazy dog 1234567890."
    file_bytes = sample.encode("ascii")
    assert decode_text_bytes(file_bytes, "ascii.txt") == sample

    pages = await parse_document(file_bytes, "ascii.txt")
    assert len(pages) == 1
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_valid_utf8():
    sample = (
        "UTF-8 document: élève, café, naïve, 🚀 rocket emoji, "
        "Japanese: こんにちは, and math: ∑(x_i) = 42."
    )
    file_bytes = sample.encode("utf-8")
    assert decode_text_bytes(file_bytes, "utf8.txt") == sample

    pages = await parse_document(file_bytes, "utf8.md")
    assert len(pages) == 1
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_utf8_with_bom():
    sample = "Document content with UTF-8 BOM. No leading BOM character should remain."
    file_bytes = codecs.BOM_UTF8 + sample.encode("utf-8")

    decoded = decode_text_bytes(file_bytes, "utf8_bom.txt")
    assert decoded == sample
    assert not decoded.startswith("\ufeff")

    pages = await parse_document(file_bytes, "utf8_bom.txt")
    assert len(pages) == 1
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_utf16_le_and_be_with_bom():
    sample = "UTF-16 document with BOM: café, déjà vu, and math symbols."

    # UTF-16 Little Endian with BOM
    le_bytes = codecs.BOM_UTF16_LE + sample.encode("utf-16-le")
    decoded_le = decode_text_bytes(le_bytes, "utf16_le.txt")
    assert decoded_le == sample
    assert not decoded_le.startswith("\ufeff")

    pages_le = await parse_document(le_bytes, "utf16_le.txt")
    assert pages_le[0]["text"] == sample

    # UTF-16 Big Endian with BOM
    be_bytes = codecs.BOM_UTF16_BE + sample.encode("utf-16-be")
    decoded_be = decode_text_bytes(be_bytes, "utf16_be.txt")
    assert decoded_be == sample
    assert not decoded_be.startswith("\ufeff")

    pages_be = await parse_document(be_bytes, "utf16_be.txt")
    assert pages_be[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_utf32_le_and_be_with_bom():
    sample = "UTF-32 document with 4-byte BOM: café and symbols."

    # UTF-32 Little Endian with BOM
    le_bytes = codecs.BOM_UTF32_LE + sample.encode("utf-32-le")
    decoded_le = decode_text_bytes(le_bytes, "utf32_le.txt")
    assert decoded_le == sample
    assert not decoded_le.startswith("\ufeff")

    pages_le = await parse_document(le_bytes, "utf32_le.txt")
    assert pages_le[0]["text"] == sample

    # UTF-32 Big Endian with BOM
    be_bytes = codecs.BOM_UTF32_BE + sample.encode("utf-32-be")
    decoded_be = decode_text_bytes(be_bytes, "utf32_be.txt")
    assert decoded_be == sample
    assert not decoded_be.startswith("\ufeff")

    pages_be = await parse_document(be_bytes, "utf32_be.txt")
    assert pages_be[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_latin1_accented_characters():
    sample = "Café crème, résumé, déjà vu, façade, naïve, tête-à-tête, año nuevo y el niño."
    file_bytes = sample.encode("latin-1")

    # Strict UTF-8 will fail for this byte sequence
    with pytest.raises(UnicodeDecodeError):
        file_bytes.decode("utf-8")

    decoded = decode_text_bytes(file_bytes, "latin1_doc.txt")
    assert decoded == sample
    assert "é" in decoded and "à" in decoded and "ñ" in decoded and "ï" in decoded

    pages = await parse_document(file_bytes, "latin1_doc.txt")
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_windows_1252_special_characters():
    sample = "“Smart quotes”, em-dash —, en-dash –, bullet •, and euro €100."
    file_bytes = sample.encode("cp1252")

    # Strict UTF-8 will fail for cp1252 quotes/euro
    with pytest.raises(UnicodeDecodeError):
        file_bytes.decode("utf-8")

    decoded = decode_text_bytes(file_bytes, "win1252_doc.txt")
    assert decoded == sample
    assert "“" in decoded and "”" in decoded and "—" in decoded and "€" in decoded

    pages = await parse_document(file_bytes, "win1252_doc.txt")
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_decode_utf16_without_bom():
    sample = "UTF-16 Little Endian without any BOM marker."
    file_bytes = sample.encode("utf-16-le")

    decoded = decode_text_bytes(file_bytes, "utf16_no_bom.txt")
    assert decoded == sample

    pages = await parse_document(file_bytes, "utf16_no_bom.txt")
    assert pages[0]["text"] == sample


@pytest.mark.asyncio
async def test_invalid_utf8_does_not_silently_drop_bytes():
    # Corrupt sequence that would have lost bytes under errors="ignore"
    corrupt_bytes = b"Start of section: \xff\xfe\x99\x88 ... rest of section."
    decoded = decode_text_bytes(corrupt_bytes, "corrupt.txt")

    # Must preserve content and not silently discard
    assert "Start of section:" in decoded
    assert "rest of section." in decoded
    # Should not be shorter than the known ASCII portions
    assert len(decoded) >= len("Start of section:  ... rest of section.")


@pytest.mark.asyncio
async def test_fallback_when_detection_fails_logs_warning_and_uses_replacement(caplog):
    # Completely unclassifiable high-chaos binary payload
    garbage_bytes = b"\x80\x81\x82\xff\xfe\x00\x01\x02\x03\x04\x05\x06\x07"

    with caplog.at_level(logging.WARNING):
        decoded = decode_text_bytes(garbage_bytes, "unidentifiable.txt")

    # Must contain Unicode replacement character \ufffd and NOT silently drop bytes
    assert "\ufffd" in decoded
    assert len(decoded) > 0

    # Must log an appropriate warning
    warning_logged = any(
        "Encoding detection failed" in record.message or "insufficient confidence" in record.message
        for record in caplog.records
    )
    assert warning_logged, f"Expected warning was not logged. Records: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_malformed_bom_fallback_and_warning(caplog):
    # Truncated UTF-16 stream with BOM (odd byte length)
    truncated_utf16 = codecs.BOM_UTF16_LE + b"A\x00B\x00\xff"

    with caplog.at_level(logging.WARNING):
        decoded = decode_text_bytes(truncated_utf16, "truncated_utf16.txt")

    assert "\ufffd" in decoded
    assert "AB" in decoded

    warning_logged = any(
        "Failed strict UTF-16 decoding" in record.message
        for record in caplog.records
    )
    assert warning_logged, f"Expected UTF-16 warning not logged: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_parse_document_json_and_md_formats():
    json_content = '{"title": "Test JSON", "accent": "crème brûlée"}'
    md_content = "# Heading\n\n- Item with quote: “test”"

    json_pages = await parse_document(json_content.encode("utf-8"), "config.json")
    assert json_pages == [{"page_number": 1, "text": json_content}]

    md_pages = await parse_document(md_content.encode("cp1252"), "notes.md")
    assert md_pages == [{"page_number": 1, "text": md_content}]


@pytest.mark.asyncio
async def test_unsupported_binary_file_fails_fast():
    # An unrecognized binary format (.bin) should continue raising ValueError
    with pytest.raises(ValueError, match=r"Unsupported binary file format: \.bin"):
        await parse_document(b"\x00\x01\x02\x03\x04\x05\xff\xfe", "archive.bin")
