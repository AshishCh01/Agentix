"""
Automated unit and integration tests for the Retrieval Evaluation Framework (Issue #33).

All tests use mocks and isolated fixtures without external APIs or live database mutations.
"""

import json
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.eval_retrieval import (
    load_dataset,
    validate_dataset,
    calculate_metrics,
    generate_markdown_report,
    export_results,
    evaluate_query,
    fetch_retrieval_candidates,
)


@pytest.fixture
def sample_dataset_dict():
    return {
        "metadata": {
            "version": "1.0",
            "evaluation_scope": {
                "knowledge_base_id": "54335692-b9bd-4693-b074-5b536bdd51af",
                "knowledge_base_name": "Test Knowledge Base",
                "document_id": "607a4e68-8135-4686-a60b-5970d65fbada",
                "document_filename": "Test_Doc.docx",
            },
        },
        "questions": [
            {
                "id": "Q01",
                "question": "What is grid computing middleware?",
                "answerable": True,
                "expected_document_id": "607a4e68-8135-4686-a60b-5970d65fbada",
                "expected_chunk_ids": ["chunk-uuid-001"],
                "expected_page": 1,
                "source_passage": "Grid computing relies on middleware such as Globus Toolkit.",
                "topic": "Grid Computing",
            },
            {
                "id": "Q02",
                "question": "How does cloud computing convert CapEx to OpEx?",
                "answerable": True,
                "expected_document_id": "607a4e68-8135-4686-a60b-5970d65fbada",
                "expected_chunk_ids": ["chunk-uuid-002"],
                "expected_page": 1,
                "source_passage": "Converts capital expenditure into operational expenditure.",
                "topic": "CapEx to OpEx",
            },
            {
                "id": "Q03",
                "question": "What are the stages of Canny edge detection?",
                "answerable": False,
                "unanswerable_category": "off_topic",
                "reason": "Canny edge detection is from computer vision and not present in this corpus.",
                "topic": "Computer Vision",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Dataset Loading & Validation Tests
# ---------------------------------------------------------------------------

def test_load_dataset_success(sample_dataset_dict, tmp_path):
    dataset_file = tmp_path / "questions.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(sample_dataset_dict, f)

    loaded = load_dataset(dataset_file)
    assert len(loaded["questions"]) == 3
    assert loaded["questions"][0]["id"] == "Q01"


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("/nonexistent/path/questions.json"))


def test_validate_dataset_valid(sample_dataset_dict):
    errors = validate_dataset(sample_dataset_dict)
    assert errors == []


def test_validate_dataset_missing_questions():
    errors = validate_dataset({})
    assert any("must contain a 'questions' list" in e for e in errors)


def test_validate_dataset_empty_questions():
    errors = validate_dataset({"questions": []})
    assert any("questions list cannot be empty" in e for e in errors)


def test_validate_dataset_duplicate_id(sample_dataset_dict):
    sample_dataset_dict["questions"].append({
        "id": "Q01",  # duplicate ID
        "question": "Another query",
        "answerable": False,
        "unanswerable_category": "off_topic",
        "reason": "duplicate",
    })
    errors = validate_dataset(sample_dataset_dict)
    assert any("Duplicate question id 'Q01'" in e for e in errors)


def test_validate_dataset_missing_expected_chunks(sample_dataset_dict):
    sample_dataset_dict["questions"][0]["expected_chunk_ids"] = []
    errors = validate_dataset(sample_dataset_dict)
    assert any("must provide non-empty 'expected_chunk_ids'" in e for e in errors)


def test_validate_dataset_missing_unanswerable_category(sample_dataset_dict):
    del sample_dataset_dict["questions"][2]["unanswerable_category"]
    errors = validate_dataset(sample_dataset_dict)
    assert any("missing 'unanswerable_category'" in e for e in errors)


# ---------------------------------------------------------------------------
# Metric Calculation Tests
# ---------------------------------------------------------------------------

def test_calculate_metrics_hit_rate():
    eval_results = [
        {
            "id": "Q01",
            "answerable": True,
            "strategies": {
                "dense_only": {"hit": True, "retrieved": []},
                "sparse_only": {"hit": False, "retrieved": []},
                "hybrid_rrf": {"hit": True, "retrieved": []},
                "hybrid_rerank": {"hit": True, "retrieved": []},
            },
        },
        {
            "id": "Q02",
            "answerable": True,
            "strategies": {
                "dense_only": {"hit": True, "retrieved": []},
                "sparse_only": {"hit": True, "retrieved": []},
                "hybrid_rrf": {"hit": True, "retrieved": []},
                "hybrid_rerank": {"hit": True, "retrieved": []},
            },
        },
        {
            "id": "Q03",
            "answerable": False,  # Unanswerable - MUST NOT inflate hit count
            "unanswerable_category": "off_topic",
            "strategies": {
                "dense_only": {"hit": False, "retrieved": [{"dense_similarity": 0.45, "filename": "doc.docx", "chunk_index": 1}]},
                "sparse_only": {"hit": False, "retrieved": []},
                "hybrid_rrf": {"hit": False, "retrieved": []},
                "hybrid_rerank": {"hit": False, "retrieved": [{"rerank_score": 0.05, "filename": "doc.docx", "chunk_index": 1}]},
            },
        },
    ]

    metrics = calculate_metrics(eval_results)
    assert metrics["total_questions"] == 3
    assert metrics["total_answerable"] == 2
    assert metrics["total_unanswerable"] == 1

    # Dense: 2/2 = 1.0 (100%)
    assert metrics["strategies"]["dense_only"]["hits"] == 2
    assert metrics["strategies"]["dense_only"]["hit_rate_at_4"] == 1.0

    # Sparse: 1/2 = 0.5 (50%)
    assert metrics["strategies"]["sparse_only"]["hits"] == 1
    assert metrics["strategies"]["sparse_only"]["hit_rate_at_4"] == 0.5
    assert metrics["strategies"]["sparse_only"]["missed_question_ids"] == ["Q01"]

    # Hybrid: 2/2 = 1.0
    assert metrics["strategies"]["hybrid_rrf"]["hits"] == 2
    assert metrics["strategies"]["hybrid_rerank"]["hits"] == 2


def test_calculate_metrics_zero_answerable():
    eval_results = [
        {
            "id": "Q01",
            "answerable": False,
            "unanswerable_category": "off_topic",
            "strategies": {
                "dense_only": {"hit": False, "retrieved": []},
                "sparse_only": {"hit": False, "retrieved": []},
                "hybrid_rrf": {"hit": False, "retrieved": []},
                "hybrid_rerank": {"hit": False, "retrieved": []},
            },
        }
    ]
    metrics = calculate_metrics(eval_results)
    assert metrics["total_answerable"] == 0
    assert metrics["strategies"]["dense_only"]["hit_rate_at_4"] == 0.0


def test_calculate_metrics_reranking_comparison():
    eval_results = [
        {
            "id": "Q01",
            "answerable": True,
            "strategies": {
                "dense_only": {"hit": False, "retrieved": []},
                "sparse_only": {"hit": False, "retrieved": []},
                "hybrid_rrf": {"hit": False, "retrieved": []},
                "hybrid_rerank": {"hit": True, "retrieved": []},  # Improved by rerank
            },
        },
        {
            "id": "Q02",
            "answerable": True,
            "strategies": {
                "dense_only": {"hit": True, "retrieved": []},
                "sparse_only": {"hit": True, "retrieved": []},
                "hybrid_rrf": {"hit": True, "retrieved": []},
                "hybrid_rerank": {"hit": False, "retrieved": []},  # Worsened by rerank
            },
        },
    ]
    metrics = calculate_metrics(eval_results)
    comp = metrics["reranking_comparison"]
    assert comp["improved_by_rerank"] == ["Q01"]
    assert comp["worsened_by_rerank"] == ["Q02"]
    assert comp["net_change"] == 0


# ---------------------------------------------------------------------------
# Retrieval Strategy Execution Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_query_hit_matching():
    query_item = {
        "id": "Q01",
        "question": "What is grid computing middleware?",
        "answerable": True,
        "expected_chunk_ids": ["target-chunk-123"],
    }

    mock_candidates = [
        {
            "chunk_id": "other-chunk-999",
            "document_id": "doc-1",
            "chunk_index": 0,
            "page_number": 1,
            "content": "Irrelevant intro content",
            "filename": "doc.docx",
            "rrf_score": 0.030,
            "sparse_rank": 3,
            "dense_rank": 2,
            "dense_similarity": 0.70,
            "source_type": "hybrid",
            "rerank_score": None,
            "rerank_logit": None,
        },
        {
            "chunk_id": "target-chunk-123",
            "document_id": "doc-1",
            "chunk_index": 1,
            "page_number": 1,
            "content": "Globus Toolkit middleware for grid computing",
            "filename": "doc.docx",
            "rrf_score": 0.033,
            "sparse_rank": 1,
            "dense_rank": 1,
            "dense_similarity": 0.88,
            "source_type": "hybrid",
            "rerank_score": None,
            "rerank_logit": None,
        },
    ]

    mock_db = AsyncMock()

    with patch("scripts.eval_retrieval.fetch_retrieval_candidates", return_value=(mock_candidates, [0.1] * 2000)), \
         patch("scripts.eval_retrieval.embedding_service.rerank_chunks", side_effect=lambda q, c: c):
        res = await evaluate_query(
            db=mock_db,
            query_item=query_item,
            knowledge_base_id="54335692-b9bd-4693-b074-5b536bdd51af",
            top_k=4,
        )

    assert res["id"] == "Q01"
    assert res["strategies"]["dense_only"]["hit"] is True
    assert res["strategies"]["sparse_only"]["hit"] is True
    assert res["strategies"]["hybrid_rrf"]["hit"] is True
    assert res["strategies"]["hybrid_rerank"]["hit"] is True

    # Check ranking and score preservation
    dense_ret = res["strategies"]["dense_only"]["retrieved"]
    assert dense_ret[0]["chunk_id"] == "target-chunk-123"
    assert dense_ret[0]["dense_rank"] == 1
    assert dense_ret[0]["strategy_rank"] == 1
    assert dense_ret[0]["is_hit"] is True


@pytest.mark.asyncio
async def test_evaluate_query_empty_candidates():
    query_item = {
        "id": "Q_EMPTY",
        "question": "A query that returns no candidates",
        "answerable": True,
        "expected_chunk_ids": ["some-chunk-id"],
    }

    mock_db = AsyncMock()
    with patch("scripts.eval_retrieval.fetch_retrieval_candidates", return_value=([], [0.0] * 2000)):
        res = await evaluate_query(
            db=mock_db,
            query_item=query_item,
            knowledge_base_id="54335692-b9bd-4693-b074-5b536bdd51af",
            top_k=4,
        )

    for strat in ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank"]:
        assert res["strategies"][strat]["hit"] is False
        assert len(res["strategies"][strat]["retrieved"]) == 0


@pytest.mark.asyncio
async def test_scope_isolation_sql_execution():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.mappings.return_value = []
    mock_db.execute.return_value = mock_result

    target_kb = "54335692-b9bd-4693-b074-5b536bdd51af"

    with patch("scripts.eval_retrieval.embedding_service.generate_embedding", return_value=[0.05] * 2000):
        candidates, vec = await fetch_retrieval_candidates(
            db=mock_db,
            query_text="Sample query",
            knowledge_base_id=target_kb,
        )

    # Verify db.execute was called and received scope_id parameter
    calls = mock_db.execute.call_args_list
    # The last call is the main hybrid SQL
    sql_call = calls[-1]
    params = sql_call[0][1]
    assert params["scope_id"] == target_kb
    assert params["list_size"] == 30
    assert params["rrf_k"] == 60


# ---------------------------------------------------------------------------
# Report & File Export Tests
# ---------------------------------------------------------------------------

def test_markdown_report_formatting():
    summary = {
        "timestamp": "2026-10-31T12:00:00Z",
        "total_questions": 2,
        "total_answerable": 1,
        "total_unanswerable": 1,
        "strategies": {
            "dense_only": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "sparse_only": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "hybrid_rrf": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "hybrid_rerank": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
        },
        "reranking_comparison": {
            "improved_by_rerank": [],
            "worsened_by_rerank": [],
            "net_change": 0,
        },
        "unanswerable_analysis": [
            {
                "id": "Q02",
                "question": "Unanswerable query",
                "category": "off_topic",
                "max_dense_similarity": 0.45,
                "max_rerank_score": 0.05,
            }
        ],
    }

    eval_results = [
        {
            "id": "Q01",
            "question": "Valid query",
            "answerable": True,
            "strategies": {
                "dense_only": {"hit": True},
                "sparse_only": {"hit": True},
                "hybrid_rrf": {"hit": True},
                "hybrid_rerank": {"hit": True},
            },
        }
    ]

    metadata = {
        "evaluation_scope": {
            "knowledge_base_name": "Test KB",
            "knowledge_base_id": "test-kb-id",
            "document_filename": "test.docx",
            "document_id": "test-doc-id",
        }
    }

    report = generate_markdown_report(summary, eval_results, metadata)
    assert "# Agentix Retrieval Evaluation Report" in report
    assert "Hit Rate@4" in report
    assert "| **Dense only**" in report
    assert "100.0%" in report
    assert "`off_topic`" in report


def test_export_results_creates_files(tmp_path):
    summary = {
        "timestamp": "2026-10-31T12:00:00Z",
        "total_questions": 1,
        "total_answerable": 1,
        "total_unanswerable": 0,
        "strategies": {
            "dense_only": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "sparse_only": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "hybrid_rrf": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
            "hybrid_rerank": {"hits": 1, "total": 1, "hit_rate_percentage": 100.0, "missed_question_ids": []},
        },
        "reranking_comparison": {"improved_by_rerank": [], "worsened_by_rerank": [], "net_change": 0},
        "unanswerable_analysis": [],
    }
    eval_results = []
    metadata = {}

    summary_file, detailed_file, report_file = export_results(
        summary=summary,
        eval_results=eval_results,
        metadata=metadata,
        output_dir=tmp_path,
    )

    assert summary_file.is_file()
    assert detailed_file.is_file()
    assert report_file.is_file()
    assert summary_file.stat().st_size > 0
    assert report_file.stat().st_size > 0
