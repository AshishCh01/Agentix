import io
import uuid
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.services.chunking_service import chunk_text
from ai.services.parser_service import (
    format_excel_sheet,
    parse_document,
    serialize_dataframe_to_text,
)
from app.models.document import Document as DBDocument
from app.routes.upload import process_document_background


def create_excel_bytes(sheets_data: dict) -> bytes:
    """Helper to generate in-memory Excel (.xlsx) file bytes with multiple sheets."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, data in sheets_data.items():
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and not data:
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


# ===========================================================================
# 1. CSV files with multiple rows, columns, and large table chunking
# ===========================================================================

@pytest.mark.asyncio
async def test_csv_large_table_repeats_headers_in_every_chunk_and_preserves_order():
    """
    Verify that a large CSV spanning multiple chunks contains the relevant
    column headers in every chunk, repeats headers across chunk boundaries,
    and preserves original row order.
    """
    total_rows = 50
    df = pd.DataFrame({
        "Name": [f"Person_{i:03d}" for i in range(total_rows)],
        "Age": [20 + (i % 30) for i in range(total_rows)],
        "Department": [f"Dept_{i % 5}" for i in range(total_rows)],
        "Salary": [50000 + i * 1000 for i in range(total_rows)],
    })

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    pages = await parse_document(csv_bytes, "employees.csv")
    assert len(pages) == 1
    page_text = pages[0]["text"]

    # Verify structured serialization format: Name: Person_000 | Age: 20 | Department: Dept_0 | Salary: 50000
    assert "Name: Person_000 | Age: 20 | Department: Dept_0 | Salary: 50000" in page_text

    # Chunk with smaller chunk_size to force multiple chunks
    chunks = chunk_text(page_text, chunk_size=350, chunk_overlap=80)
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    # 1. Every chunk must contain all relevant column headers
    for idx, c in enumerate(chunks):
        content = c["content"]
        assert "Name:" in content, f"Chunk {idx} missing 'Name:' header"
        assert "Age:" in content, f"Chunk {idx} missing 'Age:' header"
        assert "Department:" in content, f"Chunk {idx} missing 'Department:' header"
        assert "Salary:" in content, f"Chunk {idx} missing 'Salary:' header"
        assert c["length"] <= 350, f"Chunk {idx} exceeded chunk_size: {c['length']}"

    # 2. Verify row order is strictly preserved within and across chunks
    extracted_row_indices = []
    for c in chunks:
        for line in c["content"].split("\n"):
            line = line.strip()
            if line.startswith("Name: Person_"):
                # Extract number
                idx_part = line.split(" | ")[0].replace("Name: Person_", "")
                extracted_row_indices.append(int(idx_part))

    # Remove consecutive duplicates caused by chunk overlap
    deduped_indices = []
    for idx_num in extracted_row_indices:
        if not deduped_indices or deduped_indices[-1] != idx_num:
            deduped_indices.append(idx_num)

    # Must be strictly increasing from 0 up to total_rows - 1
    assert deduped_indices == list(range(total_rows))


# ===========================================================================
# 2. XLSX files with multiple sheets spanning multiple chunks
# ===========================================================================

@pytest.mark.asyncio
async def test_xlsx_multisheet_large_tables_repeat_sheet_and_column_headers():
    """
    Verify that an XLSX file with multiple sheets spanning multiple chunks
    repeats both the sheet header ('Sheet: <name>') and column headers
    across all chunk boundaries in every worksheet.
    """
    num_rows = 40
    sheets_data = {
        "Engineering": {
            "EmployeeID": [f"ENG-{i:03d}" for i in range(num_rows)],
            "Role": ["Software Engineer" if i % 2 == 0 else "Site Reliability Engineer" for i in range(num_rows)],
            "SkillLevel": [f"Level-{1 + (i % 5)}" for i in range(num_rows)],
        },
        "Marketing": {
            "CampaignID": [f"CMP-{i:03d}" for i in range(num_rows)],
            "Budget": [10000 + i * 500 for i in range(num_rows)],
            "Channel": ["Social Media" if i % 2 == 0 else "Search Ads" for i in range(num_rows)],
        },
    }

    file_bytes = create_excel_bytes(sheets_data)
    pages = await parse_document(file_bytes, "company_operations.xlsx")
    assert len(pages) == 2

    # Sheet 1: Engineering
    eng_chunks = chunk_text(pages[0]["text"], chunk_size=400, chunk_overlap=100)
    assert len(eng_chunks) >= 3, f"Expected multiple chunks for Engineering, got {len(eng_chunks)}"

    for idx, c in enumerate(eng_chunks):
        content = c["content"]
        assert content.startswith("Sheet: Engineering"), f"Chunk {idx} must start with 'Sheet: Engineering'"
        assert "EmployeeID:" in content, f"Chunk {idx} missing 'EmployeeID:' header"
        assert "Role:" in content, f"Chunk {idx} missing 'Role:' header"
        assert "SkillLevel:" in content, f"Chunk {idx} missing 'SkillLevel:' header"
        assert c["length"] <= 400

    # Sheet 2: Marketing
    mkt_chunks = chunk_text(pages[1]["text"], chunk_size=400, chunk_overlap=100)
    assert len(mkt_chunks) >= 3, f"Expected multiple chunks for Marketing, got {len(mkt_chunks)}"

    for idx, c in enumerate(mkt_chunks):
        content = c["content"]
        assert content.startswith("Sheet: Marketing"), f"Chunk {idx} must start with 'Sheet: Marketing'"
        assert "CampaignID:" in content, f"Chunk {idx} missing 'CampaignID:' header"
        assert "Budget:" in content, f"Chunk {idx} missing 'Budget:' header"
        assert "Channel:" in content, f"Chunk {idx} missing 'Channel:' header"
        assert c["length"] <= 400


# ===========================================================================
# 3. Empty cells, missing values (NaN/None), and whitespace pruning
# ===========================================================================

@pytest.mark.asyncio
async def test_table_missing_values_and_empty_cells():
    """
    Verify that missing values (NaN, None, empty string) are handled safely
    without data loss, and phantom Unnamed columns with all NaNs are pruned.
    """
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Ashish", None, "Bob"],
        "Score": [98.5, np.nan, 85.0],
        "Notes": ["", "  ", "Top performer"],
        "Unnamed: 4": [np.nan, np.nan, np.nan],  # Phantom empty column
    })

    text = serialize_dataframe_to_text(df, sheet_name="Results")
    assert "Sheet: Results" in text

    # Check row 1: empty string serialized safely as NaN
    assert "ID: 1 | Name: Ashish | Score: 98.5 | Notes: NaN" in text

    # Check row 2: None, NaN, and whitespace serialized safely as NaN
    assert "ID: 2 | Name: NaN | Score: NaN | Notes: NaN" in text

    # Check row 3: intact data preserved
    assert "ID: 3 | Name: Bob | Score: 85.0 | Notes: Top performer" in text

    # Phantom column should not be emitted
    assert "Unnamed: 4" not in text


@pytest.mark.asyncio
async def test_table_formatting_prunes_whitespace_and_handles_embedded_newlines():
    """
    Verify that unnecessary whitespace is removed, multiline cell values
    are flattened to avoid breaking row structure, and pipes are escaped.
    """
    df = pd.DataFrame({
        "Project": ["  Alpha Core  ", "Beta | System"],
        "Description": ["High performance\nmultithreaded engine", "Cloud\r\nStorage"],
    })

    text = serialize_dataframe_to_text(df)
    lines = text.split("\n")
    assert len(lines) == 2, f"Expected exactly 2 lines for 2 rows, got {len(lines)}"

    # First row: whitespace stripped, embedded newline flattened to space
    assert "Project: Alpha Core | Description: High performance multithreaded engine" == lines[0]

    # Second row: pipe in value escaped
    assert "Project: Beta \\| System | Description: Cloud Storage" == lines[1]


# ===========================================================================
# 4. Correct association between column names and values
# ===========================================================================

@pytest.mark.asyncio
async def test_column_value_association():
    """
    Verify strict association between each column name and its respective value.
    """
    df = pd.DataFrame({
        "First Name": ["Ashish"],
        "Age": [21],
        "Department": ["Data Science"],
        "Active": [True],
    })

    text = serialize_dataframe_to_text(df)
    expected = "First Name: Ashish | Age: 21 | Department: Data Science | Active: True"
    assert text == expected


# ===========================================================================
# 5. Embedding pipeline integration verification
# ===========================================================================

@pytest.mark.asyncio
async def test_csv_serialized_content_reaches_embedding_pipeline():
    """
    Verify that structured tabular serialization reaches the embedding pipeline
    and database chunks correctly during background document processing.
    """
    doc_id = uuid.uuid4()
    total_rows = 30
    df = pd.DataFrame({
        "Metric": [f"M_{i}" for i in range(total_rows)],
        "Value": [i * 10 for i in range(total_rows)],
        "Status": ["PASS" if i % 2 == 0 else "FAIL" for i in range(total_rows)],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    mock_doc = MagicMock(spec=DBDocument)
    mock_doc.id = doc_id
    mock_doc.status = "processing"
    mock_doc.ocr_warning_count = 0
    mock_doc.error_message = None

    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt).lower()
        if "count(" in stmt_str:
            mock_res.scalar.return_value = len(mock_session.add.call_args_list)
        else:
            mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session.execute.side_effect = mock_execute

    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.embedding_service") as mock_es:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        
        # Capture texts passed to embedding service
        captured_texts = []
        async def mock_batch_embeddings(texts, **kwargs):
            captured_texts.extend(texts)
            return [[0.05] * 1536 for _ in texts]

        mock_es.generate_batch_embeddings.side_effect = mock_batch_embeddings

        await process_document_background(doc_id, csv_bytes, "metrics.csv", "path/metrics.csv")

        # 1. Verify embedding service received chunks containing column headers
        assert len(captured_texts) >= 1
        for text in captured_texts:
            assert "Metric:" in text
            assert "Value:" in text
            assert "Status:" in text

        # 2. Verify database chunk records contain headers and proper values
        add_calls = mock_session.add.call_args_list
        assert len(add_calls) >= 1
        for call in add_calls:
            chunk = call[0][0]
            assert "Metric:" in chunk.content
            assert "Value:" in chunk.content
            assert "Status:" in chunk.content
            assert chunk.document_id == doc_id

        assert mock_doc.status == "ready"
        mock_session.commit.assert_called()


# ===========================================================================
# 6. Non-tabular formats preserve existing parsing and chunking behavior
# ===========================================================================

@pytest.mark.asyncio
async def test_non_tabular_formats_unaffected():
    """
    Verify that text documents (txt, md, json) are not affected by tabular logic
    and continue to chunk normally.
    """
    text_content = "This is a standard text document.\n\nIt has multiple paragraphs.\n\nEach paragraph discusses a topic."
    pages = await parse_document(text_content.encode("utf-8"), "document.txt")

    assert len(pages) == 1
    assert "standard text document" in pages[0]["text"]
    assert "Sheet:" not in pages[0]["text"]

    chunks = chunk_text(pages[0]["text"], chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    # Check that chunks do not invent Sheet or table prefixes
    for c in chunks:
        assert not c["content"].startswith("Sheet:")
