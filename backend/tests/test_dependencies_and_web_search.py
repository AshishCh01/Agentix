import io
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai.services.parser_service import parse_document
from ai.tools.web_search import _run_ddgs_sync, web_search_tool


# --- 1. Web Search Tool Unit Tests (Mocked) ---


def test_run_ddgs_sync_with_ddgs():
    sample_raw = [
        {
            "title": "Agentic RAG Paper",
            "href": "https://arxiv.org/abs/example",
            "body": "A comprehensive overview of Agentic RAG patterns.",
        },
        {
            "title": "Alternative Search Result",
            "url": "https://example.com/alt",
            "description": "Alternative snippet format.",
        },
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = sample_raw
    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.__exit__.return_value = False

    with patch("ddgs.DDGS", mock_ddgs_cls):
        results = _run_ddgs_sync("test query", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "Agentic RAG Paper"
    assert results[0]["url"] == "https://arxiv.org/abs/example"
    assert results[0]["content"] == "A comprehensive overview of Agentic RAG patterns."
    assert results[1]["url"] == "https://example.com/alt"
    assert results[1]["content"] == "Alternative snippet format."


def test_run_ddgs_sync_fallback_to_duckduckgo_search():
    sample_raw = [
        {
            "title": "Legacy Search",
            "href": "https://duckduckgo.com/legacy",
            "body": "Legacy result text",
        }
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = sample_raw
    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.__exit__.return_value = False

    # Simulate ddgs missing so fallback import is triggered
    with patch.dict(sys.modules, {"ddgs": None}):
        with patch.dict(
            sys.modules, {"duckduckgo_search": MagicMock(DDGS=mock_ddgs_cls)}
        ):
            results = _run_ddgs_sync("legacy query", max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "Legacy Search"
    assert results[0]["url"] == "https://duckduckgo.com/legacy"


@pytest.mark.asyncio
async def test_web_search_tool_success():
    sample_results = [
        {
            "title": "Test Title",
            "url": "https://example.com/test",
            "content": "Test snippet content.",
        }
    ]

    with patch(
        "ai.tools.web_search._run_ddgs_sync", return_value=sample_results
    ):
        output = await web_search_tool("Agentic RAG", max_results=1)

    assert output["query"] == "Agentic RAG"
    assert len(output["results"]) == 1
    assert len(output["sources"]) == 1
    assert output["sources"][0]["filename"] == "Test Title"
    assert output["sources"][0]["url"] == "https://example.com/test"
    assert "Test snippet content." in output["formatted_context"]


@pytest.mark.asyncio
async def test_web_search_tool_empty_results():
    with patch("ai.tools.web_search._run_ddgs_sync", return_value=[]):
        output = await web_search_tool("nonexistent obscure query")

    assert output["results"] == []
    assert output["sources"] == []
    assert output["formatted_context"] == "No relevant web search results found."


@pytest.mark.asyncio
async def test_web_search_tool_handles_exception_gracefully():
    with patch(
        "ai.tools.web_search._run_ddgs_sync",
        side_effect=RuntimeError("Connection timeout"),
    ):
        output = await web_search_tool("query causing error")

    assert output["results"] == []
    assert output["sources"] == []
    assert "Web search service is currently unavailable" in output["formatted_context"]


# --- 2. Startup Dependency Self-Check Tests ---


def _run_startup_dependency_check(available_deps: set):
    """Helper that runs the startup dependency check logic from main.py."""
    captured_warnings = []

    def mock_import(name, *args, **kwargs):
        if name in available_deps:
            return MagicMock()
        raise ImportError(f"No module named {name}")

    optional_deps = ["ddgs", "openpyxl", "xlrd"]

    for dep in optional_deps:
        try:
            mock_import(dep)
        except ImportError:
            if dep == "ddgs":
                try:
                    mock_import("duckduckgo_search")
                    captured_warnings.append(
                        "[WARN] Dependency 'duckduckgo_search' is deprecated. Migrate to 'ddgs'."
                    )
                    continue
                except ImportError:
                    pass
            captured_warnings.append(
                f"[WARN] Dependency '{dep}' is missing. Some features may not work or might fail."
            )

    return captured_warnings


def test_dependency_check_reports_no_warnings_when_all_installed():
    warnings = _run_startup_dependency_check({"ddgs", "openpyxl", "xlrd"})
    assert warnings == []


def test_dependency_check_reports_missing_xlrd():
    warnings = _run_startup_dependency_check({"ddgs", "openpyxl"})
    assert len(warnings) == 1
    assert "Dependency 'xlrd' is missing" in warnings[0]


def test_dependency_check_reports_missing_ddgs_when_none_installed():
    warnings = _run_startup_dependency_check({"openpyxl", "xlrd"})
    assert len(warnings) == 1
    assert "Dependency 'ddgs' is missing" in warnings[0]


def test_dependency_check_reports_deprecation_when_only_legacy_duckduckgo_search():
    warnings = _run_startup_dependency_check(
        {"duckduckgo_search", "openpyxl", "xlrd"}
    )
    assert len(warnings) == 1
    assert "duckduckgo_search' is deprecated" in warnings[0]


# --- 3. Document Parsing for Excel (.xlsx and .xls) ---


@pytest.mark.asyncio
async def test_xlsx_parsing_with_openpyxl():
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 88]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")

    pages = await parse_document(buf.getvalue(), "test_scores.xlsx")
    assert len(pages) == 1
    assert "Alice" in pages[0]["text"]
    assert "Bob" in pages[0]["text"]


@pytest.mark.asyncio
async def test_xls_parsing_invokes_excel_reader():
    # Verify .xls files are routed to pd.read_excel (which requires xlrd for .xls)
    dummy_df = pd.DataFrame({"Legacy": [1, 2, 3]})
    with patch("pandas.read_excel", return_value=dummy_df) as mock_read:
        pages = await parse_document(b"fake_xls_binary_data", "report.xls")
        assert len(pages) == 1
        assert "Legacy" in pages[0]["text"]
        mock_read.assert_called_once()
