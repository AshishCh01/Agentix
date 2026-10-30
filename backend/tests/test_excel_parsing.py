import io
import uuid
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.services.chunking_service import chunk_text
from ai.services.parser_service import format_excel_sheet, parse_document
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
# GROUP 1: Multi-Sheet Workbook Extraction (Issue #20 Core Fix)
# ===========================================================================

@pytest.mark.asyncio
async def test_two_sheet_workbook_extracts_both_sheets():
    """
    Verify that a two-sheet workbook extracts both sheets, preserving
    each worksheet's name and its tabular content.
    Issue #20: pd.read_excel previously only read the default sheet.
    """
    sheets = {
        "Q1_Sales": {
            "Product": ["Laptop Pro", "Wireless Mouse"],
            "Units": [150, 400],
            "Revenue": [180000, 10000],
        },
        "Q2_Sales": {
            "Product": ["Laptop Pro", "Mechanical Keyboard"],
            "Units": [220, 310],
            "Revenue": [264000, 31000],
        },
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "quarterly_reports.xlsx")

    # Critical check: exactly 2 pages returned (one per sheet)
    assert len(pages) == 2

    # Sheet 1 assertions
    assert pages[0]["page_number"] == 1
    assert "Sheet: Q1_Sales" in pages[0]["text"]
    assert "Laptop Pro" in pages[0]["text"]
    assert "180000" in pages[0]["text"]

    # Sheet 2 assertions
    assert pages[1]["page_number"] == 2
    assert "Sheet: Q2_Sales" in pages[1]["text"]
    assert "Mechanical Keyboard" in pages[1]["text"]
    assert "264000" in pages[1]["text"]


@pytest.mark.asyncio
async def test_three_sheet_workbook_order_and_names_preserved():
    """
    Verify that a workbook with three sheets preserves the exact sheet order
    and maps each to sequential page numbers with correct sheet titles.
    """
    sheets = {
        "Employees": {"ID": [101, 102], "Name": ["Alice", "Bob"]},
        "Departments": {"DeptID": ["ENG", "OPS"], "DeptName": ["Engineering", "Operations"]},
        "Budgets": {"DeptID": ["ENG", "OPS"], "Budget": [1500000, 800000]},
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "company_data.xlsx")

    assert len(pages) == 3
    assert pages[0]["page_number"] == 1
    assert "Sheet: Employees" in pages[0]["text"]
    assert "Alice" in pages[0]["text"]

    assert pages[1]["page_number"] == 2
    assert "Sheet: Departments" in pages[1]["text"]
    assert "Engineering" in pages[1]["text"]

    assert pages[2]["page_number"] == 3
    assert "Sheet: Budgets" in pages[2]["text"]
    assert "1500000" in pages[2]["text"]


# ===========================================================================
# GROUP 2: Single-Sheet, Missing Values, and Empty Sheets
# ===========================================================================

@pytest.mark.asyncio
async def test_single_sheet_backward_compatibility():
    """
    Verify that single-sheet workbooks continue to parse cleanly
    with the sheet name prefix included.
    """
    sheets = {
        "Summary": {"Metric": ["Users", "Sessions"], "Count": [12000, 45000]}
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "summary.xlsx")

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Sheet: Summary" in pages[0]["text"]
    assert "Users" in pages[0]["text"]
    assert "12000" in pages[0]["text"]


@pytest.mark.asyncio
async def test_sheet_with_missing_values_nan():
    """
    Verify that sheets with missing values (None / NaN) are parsed
    without dropping rows or raising exceptions.
    """
    df = pd.DataFrame({
        "Item": ["Alpha", "Beta", np.nan],
        "Category": ["Hardware", np.nan, "Software"],
        "Score": [95.5, np.nan, 88.0],
    })
    file_bytes = create_excel_bytes({"Metrics": df})
    pages = await parse_document(file_bytes, "missing_values.xlsx")

    assert len(pages) == 1
    text = pages[0]["text"]
    assert "Sheet: Metrics" in text
    assert "Alpha" in text
    assert "Hardware" in text
    assert "Software" in text
    assert "NaN" in text


@pytest.mark.asyncio
async def test_empty_sheet_in_multisheet_workbook():
    """
    Verify that an empty sheet in a multi-sheet workbook does not break parsing
    of adjacent populated sheets and preserves the sheet name.
    """
    sheets = {
        "Populated_1": {"A": [1, 2], "B": [3, 4]},
        "Empty_Sheet": pd.DataFrame(),
        "Populated_2": {"X": ["foo", "bar"], "Y": [10, 20]},
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "mixed_empty.xlsx")

    assert len(pages) == 3
    assert "Sheet: Populated_1" in pages[0]["text"]
    assert "Sheet: Empty_Sheet" in pages[1]["text"]
    assert "Sheet: Populated_2" in pages[2]["text"]


@pytest.mark.asyncio
async def test_all_empty_sheets_workbook():
    """
    Verify that a workbook where all sheets are empty does not crash.
    """
    sheets = {
        "Blank1": pd.DataFrame(),
        "Blank2": pd.DataFrame(),
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "all_blank.xlsx")

    assert len(pages) == 2
    assert "Sheet: Blank1" in pages[0]["text"]
    assert "Sheet: Blank2" in pages[1]["text"]


@pytest.mark.asyncio
async def test_empty_bytes_and_invalid_excel():
    """
    Verify that empty bytes return a clean empty page without throwing exceptions.
    """
    empty_pages = await parse_document(b"", "empty.xlsx")
    assert empty_pages == [{"page_number": 1, "text": ""}]

    corrupt_pages = await parse_document(b"not an excel file", "corrupt.xlsx")
    assert corrupt_pages == [{"page_number": 1, "text": ""}]


# ===========================================================================
# GROUP 3: CSV Parsing Regression
# ===========================================================================

@pytest.mark.asyncio
async def test_csv_parsing_regression():
    """Verify that CSV parsing remains functional and handles empty/normal files."""
    csv_content = "Name,Age,Role\nAlice,30,Engineer\nBob,25,Designer"
    pages = await parse_document(csv_content.encode("utf-8"), "team.csv")

    assert len(pages) == 1
    assert "Alice" in pages[0]["text"]
    assert "Engineer" in pages[0]["text"]

    empty_csv_pages = await parse_document(b"", "empty.csv")
    assert empty_csv_pages == [{"page_number": 1, "text": ""}]


# ===========================================================================
# GROUP 4: Chunking and Ingestion Integration
# ===========================================================================

@pytest.mark.asyncio
async def test_excel_multisheet_chunking():
    """
    Verify that multi-sheet Excel content preserves sheet boundaries
    when processed by chunk_text.
    """
    sheets = {
        "Inventory": {
            "SKU": [f"SKU-{i:03d}" for i in range(10)],
            "Stock": [100 + i for i in range(10)],
        },
        "Suppliers": {
            "SupplierID": ["SUP-A", "SUP-B"],
            "Location": ["San Francisco", "Tokyo"],
        },
    }
    file_bytes = create_excel_bytes(sheets)
    pages = await parse_document(file_bytes, "supply_chain.xlsx")

    assert len(pages) == 2
    chunks_s1 = chunk_text(pages[0]["text"], chunk_size=600, chunk_overlap=120)
    chunks_s2 = chunk_text(pages[1]["text"], chunk_size=600, chunk_overlap=120)

    assert len(chunks_s1) >= 1
    assert "Sheet: Inventory" in chunks_s1[0]["content"]

    assert len(chunks_s2) >= 1
    assert "Sheet: Suppliers" in chunks_s2[0]["content"]


@pytest.mark.asyncio
async def test_excel_multisheet_end_to_end_ingestion():
    """
    End-to-end integration test verifying that a multi-sheet Excel file
    progresses through parsing, chunking per sheet, embedding generation,
    and database persistence with document status 'ready'.
    """
    doc_id = uuid.uuid4()
    sheets = {
        "Products": {"Item": ["Cloud Server", "CDN"], "MonthlyRate": [400, 150]},
        "Clients": {"ClientName": ["Acme Corp", "Beta LLC"], "Tier": ["Enterprise", "Standard"]},
    }
    file_bytes = create_excel_bytes(sheets)

    mock_doc = MagicMock(spec=DBDocument)
    mock_doc.id = doc_id
    mock_doc.status = "processing"
    mock_doc.ocr_warning_count = 0
    mock_doc.error_message = None

    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt).lower()
        if "count(" in stmt_str:
            mock_res.scalar.return_value = 2  # 2 chunks total
        else:
            mock_res.scalar_one_or_none.return_value = mock_doc
        return mock_res

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute.side_effect = mock_execute

    with patch("app.routes.upload.AsyncSessionLocal") as mock_session_maker, \
         patch("app.routes.upload.embedding_service") as mock_es:

        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_es.generate_batch_embeddings = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

        await process_document_background(doc_id, file_bytes, "infrastructure.xlsx", "path/infrastructure.xlsx")

        # Verify chunks were created for both sheets with respective page_numbers
        add_calls = mock_session.add.call_args_list
        assert len(add_calls) >= 2

        chunk1 = add_calls[0][0][0]
        chunk2 = add_calls[1][0][0]

        assert chunk1.page_number == 1
        assert "Sheet: Products" in chunk1.content
        assert "Cloud Server" in chunk1.content

        assert chunk2.page_number == 2
        assert "Sheet: Clients" in chunk2.content
        assert "Acme Corp" in chunk2.content

        # Verify document status transitioned to ready
        assert mock_doc.status == "ready"
        assert mock_doc.ocr_warning_count == 0
        assert mock_doc.error_message is None
        mock_session.commit.assert_called()
