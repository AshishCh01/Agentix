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
    compute_ranked_metrics,
    normalize_evidence_sets,
    annotate_candidate_hits,
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



# ---------------------------------------------------------------------------
# Multi-Document & Advanced Metric Tests (Hit Rate@1, MRR@4, nDCG@4, Per-Doc)
# ---------------------------------------------------------------------------

def test_calculate_metrics_advanced_rank_metrics():
    """Verify exact calculation of Hit Rate@1, Hit Rate@4, MRR@4, and nDCG@4."""
    eval_results = [
        # Q1: Hit at Rank 1 (RR = 1.0, nDCG = 1.0)
        {
            "id": "Q01",
            "answerable": True,
            "expected_document_id": "doc-a",
            "strategies": {
                "dense_only": {
                    "hit": True,
                    "hit_at_1": True,
                    "hit_at_4": True,
                    "first_hit_rank": 1,
                    "reciprocal_rank": 1.0,
                    "ndcg": 1.0,
                },
                "sparse_only": {"hit": False, "hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0},
                "hybrid_rrf": {"hit": True, "hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0},
                "hybrid_rerank": {"hit": True, "hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0},
            },
        },
        # Q2: Hit at Rank 2 (RR = 0.5, nDCG = 1/log2(3) = 0.6309)
        {
            "id": "Q02",
            "answerable": True,
            "expected_document_id": "doc-b",
            "strategies": {
                "dense_only": {
                    "hit": True,
                    "hit_at_1": False,
                    "hit_at_4": True,
                    "first_hit_rank": 2,
                    "reciprocal_rank": 0.5,
                    "ndcg": 0.6309,
                },
                "sparse_only": {"hit": False, "hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0},
                "hybrid_rrf": {"hit": True, "hit_at_1": False, "hit_at_4": True, "reciprocal_rank": 0.5, "ndcg": 0.6309},
                "hybrid_rerank": {"hit": True, "hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0},
            },
        },
        # Q3: Miss (not in top 4, RR = 0.0, nDCG = 0.0)
        {
            "id": "Q03",
            "answerable": True,
            "expected_document_id": "doc-a",
            "strategies": {
                "dense_only": {
                    "hit": False,
                    "hit_at_1": False,
                    "hit_at_4": False,
                    "first_hit_rank": None,
                    "reciprocal_rank": 0.0,
                    "ndcg": 0.0,
                },
                "sparse_only": {"hit": False, "hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0},
                "hybrid_rrf": {"hit": False, "hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0},
                "hybrid_rerank": {"hit": False, "hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0},
            },
        },
    ]

    metrics = calculate_metrics(eval_results)
    dense_m = metrics["strategies"]["dense_only"]
    assert dense_m["hits_at_1"] == 1
    assert dense_m["hit_rate_at_1"] == round(1 / 3, 4)
    assert dense_m["hits_at_4"] == 2
    assert dense_m["hit_rate_at_4"] == round(2 / 3, 4)
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert dense_m["mrr_at_4"] == 0.5
    # nDCG = (1.0 + 0.6309 + 0.0) / 3 = 0.5436
    assert abs(dense_m["ndcg_at_4"] - 0.5436) < 0.001


def test_calculate_metrics_per_document_breakdown():
    """Verify that per-document metrics cleanly isolate questions by document ID."""
    eval_results = [
        {
            "id": "Q01",
            "answerable": True,
            "expected_document_id": "doc-cloud",
            "strategies": {
                "dense_only": {"hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": [{"document_id": "doc-cloud", "filename": "Cloud.docx"}]},
                "sparse_only": {"hit_at_1": False, "hit_at_4": False, "reciprocal_rank": 0.0, "ndcg": 0.0, "retrieved": []},
                "hybrid_rrf": {"hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": []},
                "hybrid_rerank": {"hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": []},
            },
        },
        {
            "id": "Q02",
            "answerable": True,
            "expected_document_id": "doc-cv",
            "strategies": {
                "dense_only": {"hit_at_1": False, "hit_at_4": True, "reciprocal_rank": 0.5, "ndcg": 0.6309, "retrieved": [{"document_id": "doc-cv", "filename": "CV.docx"}]},
                "sparse_only": {"hit_at_1": False, "hit_at_4": True, "reciprocal_rank": 0.5, "ndcg": 0.6309, "retrieved": []},
                "hybrid_rrf": {"hit_at_1": False, "hit_at_4": True, "reciprocal_rank": 0.5, "ndcg": 0.6309, "retrieved": []},
                "hybrid_rerank": {"hit_at_1": True, "hit_at_4": True, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": []},
            },
        },
    ]

    metrics = calculate_metrics(eval_results)
    per_doc = metrics["per_document_metrics"]

    assert "doc-cloud" in per_doc
    assert "doc-cv" in per_doc
    assert per_doc["doc-cloud"]["total_questions"] == 1
    assert per_doc["doc-cloud"]["strategies"]["dense_only"]["hit_rate_at_1"] == 1.0
    assert per_doc["doc-cloud"]["strategies"]["dense_only"]["hit_rate_at_4"] == 1.0

    assert per_doc["doc-cv"]["total_questions"] == 1
    assert per_doc["doc-cv"]["strategies"]["dense_only"]["hit_rate_at_1"] == 0.0
    assert per_doc["doc-cv"]["strategies"]["dense_only"]["hit_rate_at_4"] == 1.0
    assert per_doc["doc-cv"]["strategies"]["hybrid_rerank"]["hit_rate_at_1"] == 1.0


def test_v2_dataset_schema_validation():
    """Verify that retrieval_questions_v2.json exists, loads, and passes validation."""
    v2_path = Path(__file__).resolve().parent.parent / "evaluation" / "retrieval_questions_v2.json"
    assert v2_path.is_file(), f"retrieval_questions_v2.json not found at {v2_path}"

    data = load_dataset(v2_path)
    errors = validate_dataset(data)
    assert errors == []

    # Verify dataset proportions: 30 total, 18 answerable (6 per doc), 12 unanswerable
    assert len(data["questions"]) == 30
    answerable = [q for q in data["questions"] if q["answerable"]]
    unanswerable = [q for q in data["questions"] if not q["answerable"]]
    assert len(answerable) == 18
    assert len(unanswerable) == 12

    # Check unanswerable breakdown (4 off-topic, 4 same-subject missing, 4 vague)
    categories = [q["unanswerable_category"] for q in unanswerable]
    assert categories.count("off_topic") == 4
    assert categories.count("same_subject_missing_detail") == 4
    assert categories.count("vague_unresolvable_followup") == 4

    # Check 3 documents represented with 6 questions each
    docs = set(q["expected_document_id"] for q in answerable)
    assert len(docs) == 3
    for d in docs:
        assert sum(1 for q in answerable if q["expected_document_id"] == d) == 6


# ---------------------------------------------------------------------------
# Multi-Chunk & Alternative Evidence Sets Tests (Audit Corrections)
# ---------------------------------------------------------------------------

def test_normalize_evidence_sets():
    """Verify normalization of evidence sets across new and backward-compatible formats."""
    # Explicit evidence sets
    sets = normalize_evidence_sets(expected_evidence_sets=[["c1", "c2"], ["c3"]])
    assert len(sets) == 2
    assert {"c1", "c2"} in sets
    assert {"c3"} in sets

    # Backward compatibility: expected_chunks set/list
    sets_compat = normalize_evidence_sets(expected_chunks=["c1", "c2"])
    assert len(sets_compat) == 2
    assert {"c1"} in sets_compat
    assert {"c2"} in sets_compat

    # Empty fallback
    assert normalize_evidence_sets() == []


def test_compute_ranked_metrics_single_chunk_sufficient():
    """Verify single-chunk evidence calculation across ranks."""
    evidence = [["c1"]]

    # Rank 1 hit
    r1 = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "c4"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r1, expected_evidence_sets=evidence)
    assert h1 is True
    assert h4 is True
    assert rank == 1
    assert rr == 1.0
    assert ndcg == 1.0

    # Rank 3 hit
    r3 = [{"chunk_id": "c0"}, {"chunk_id": "c9"}, {"chunk_id": "c1"}, {"chunk_id": "c4"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r3, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is True
    assert rank == 3
    assert abs(rr - 1 / 3) < 1e-6
    # 1 / log2(3 + 1) = 0.5
    assert abs(ndcg - 0.5) < 1e-6

    # Miss (rank > 4)
    r_miss = [{"chunk_id": "c0"}, {"chunk_id": "c9"}, {"chunk_id": "c8"}, {"chunk_id": "c7"}, {"chunk_id": "c1"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_miss, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is False
    assert rank is None
    assert rr == 0.0
    assert ndcg == 0.0


def test_compute_ranked_metrics_multi_chunk_required():
    """Verify multi-chunk evidence sets require ALL chunks to form a complete hit."""
    evidence = [["c1", "c2"]]

    # Both c1 (rank 1) and c2 (rank 3) in top 4 -> completed at rank 3
    r = [{"chunk_id": "c1"}, {"chunk_id": "c0"}, {"chunk_id": "c2"}, {"chunk_id": "c4"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is True
    assert rank == 3
    assert abs(rr - 1 / 3) < 1e-6
    assert abs(ndcg - 0.5) < 1e-6

    # Both c1 (rank 1) and c2 (rank 2) in top 4 -> completed at rank 2
    r_2 = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c0"}, {"chunk_id": "c4"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_2, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is True
    assert rank == 2
    assert rr == 0.5
    assert abs(ndcg - (1.0 / 1.5849625)) < 1e-4


def test_compute_ranked_metrics_partial_evidence_not_hit():
    """Verify that retrieving only partial evidence (subset of required chunks) is NOT a hit."""
    evidence = [["c1", "c2"]]

    # Only c1 at rank 1, c2 missing
    r_partial_1 = [{"chunk_id": "c1"}, {"chunk_id": "x1"}, {"chunk_id": "x2"}, {"chunk_id": "x3"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_partial_1, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is False
    assert rank is None
    assert rr == 0.0
    assert ndcg == 0.0

    # Only c2 at rank 4, c1 missing
    r_partial_2 = [{"chunk_id": "x1"}, {"chunk_id": "x2"}, {"chunk_id": "x3"}, {"chunk_id": "c2"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_partial_2, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is False
    assert rank is None
    assert rr == 0.0
    assert ndcg == 0.0


def test_compute_ranked_metrics_alternative_evidence_sets():
    """Verify alternative sufficient evidence sets choose the earliest completion rank."""
    # Alternative sets: Set A = [c1], Set B = [c2, c3]
    evidence = [["c1"], ["c2", "c3"]]

    # Case A: Set A satisfied at rank 2
    r_a = [{"chunk_id": "x1"}, {"chunk_id": "c1"}, {"chunk_id": "x2"}, {"chunk_id": "x3"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_a, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is True
    assert rank == 2
    assert rr == 0.5

    # Case B: Set B satisfied at rank 2 (c2 at 1, c3 at 2), Set A missing
    r_b = [{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "x1"}, {"chunk_id": "x2"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_b, expected_evidence_sets=evidence)
    assert h1 is False
    assert h4 is True
    assert rank == 2
    assert rr == 0.5

    # Case C: Both Set A and Set B completed -> chooses min completion rank
    # Set A (c1) complete at rank 4; Set B (c2, c3) complete at rank 2
    r_both = [{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "x1"}, {"chunk_id": "c1"}]
    h1, h4, rank, rr, ndcg = compute_ranked_metrics(r_both, expected_evidence_sets=evidence)
    assert rank == 2
    assert rr == 0.5


def test_dynamic_report_statistics_and_terminology():
    """Verify that report uses dynamic statistics, Binary nDCG@4 terminology, and score interpretation notes."""
    summary = {
        "timestamp": "2026-10-31T12:00:00Z",
        "total_questions": 2,
        "total_answerable": 1,
        "total_unanswerable": 1,
        "strategies": {
            "dense_only": {"hits": 1, "hits_at_1": 1, "hits_at_4": 1, "total": 1, "hit_rate_percentage": 100.0, "hit_rate_at_1_percentage": 100.0, "mrr_at_4": 1.0, "ndcg_at_4": 1.0, "missed_question_ids": []},
            "sparse_only": {"hits": 1, "hits_at_1": 1, "hits_at_4": 1, "total": 1, "hit_rate_percentage": 100.0, "hit_rate_at_1_percentage": 100.0, "mrr_at_4": 1.0, "ndcg_at_4": 1.0, "missed_question_ids": []},
            "hybrid_rrf": {"hits": 1, "hits_at_1": 1, "hits_at_4": 1, "total": 1, "hit_rate_percentage": 100.0, "hit_rate_at_1_percentage": 100.0, "mrr_at_4": 1.0, "ndcg_at_4": 1.0, "missed_question_ids": []},
            "hybrid_rerank": {"hits": 1, "hits_at_1": 1, "hits_at_4": 1, "total": 1, "hit_rate_percentage": 100.0, "hit_rate_at_1_percentage": 100.0, "mrr_at_4": 1.0, "ndcg_at_4": 1.0, "missed_question_ids": []},
        },
        "per_document_metrics": {
            "doc-1": {
                "filename": "Doc1.docx",
                "total_questions": 1,
                "strategies": {
                    "dense_only": {"hit_rate_at_1": 1.0, "hit_rate_at_4": 1.0, "mrr_at_4": 1.0},
                    "sparse_only": {"hit_rate_at_1": 1.0, "hit_rate_at_4": 1.0, "mrr_at_4": 1.0},
                    "hybrid_rrf": {"hit_rate_at_1": 1.0, "hit_rate_at_4": 1.0, "mrr_at_4": 1.0},
                    "hybrid_rerank": {"hit_rate_at_1": 1.0, "hit_rate_at_4": 1.0, "mrr_at_4": 1.0},
                },
            }
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
                "max_rerank_score": 0.000018,
            }
        ],
    }

    eval_results = [
        {
            "id": "Q01",
            "question": "Valid query",
            "answerable": True,
            "expected_document_id": "doc-1",
            "strategies": {
                "dense_only": {"hit": True, "first_hit_rank": 1},
                "sparse_only": {"hit": True, "first_hit_rank": 1},
                "hybrid_rrf": {"hit": True, "first_hit_rank": 1},
                "hybrid_rerank": {"hit": True, "first_hit_rank": 1, "retrieved": [{"chunk_id": "c1", "rerank_score": 0.9950}]},
            },
        }
    ]

    metadata = {
        "evaluation_scope": {
            "knowledge_base_name": "Test KB",
            "knowledge_base_id": "test-kb-id",
        }
    }

    report = generate_markdown_report(summary, eval_results, metadata)

    # Terminology checks
    assert "Binary nDCG@4" in report
    assert "graded position discounting" not in report

    # Score interpretation note
    assert "empirical monotonic relevance scores" in report
    assert "NOT calibrated posterior probabilities" in report

    # Balanced reranker claim
    assert "not a universal improvement across all corpus types" in report

    # Sparse FTS terminology check
    assert "Sparse PostgreSQL FTS (ts_rank_cd)" in report
    assert "Sparse BM25/FTS" not in report

    # Per-document table contains Hit@1 and MRR@4 headers
    assert "Dense Hit@1 (Hit@4)" in report
    assert "RRF MRR@4" in report
    assert "Rerank MRR@4" in report

    # Dynamic statistics (exact numbers from unanswerable_analysis and eval_results)
    assert "0.000018" in report
    assert "0.9950" in report
    assert "< 0.20" not in report


def test_v2_q14_evidence_set_validation():
    """Verify that Q14 ground-truth defines Chunk 22 as the simplified independent sufficient evidence set."""
    v2_path = Path(__file__).resolve().parent.parent / "evaluation" / "retrieval_questions_v2.json"
    data = load_dataset(v2_path)
    q14 = next(q for q in data["questions"] if q["id"] == "Q14")

    chunk_22 = "1e9c01e2-0966-4985-874c-0d46d12d83d9"
    chunk_21 = "1940a3fe-484e-49c3-bbdb-be2524d0dc10"
    # Chunk 22 independently satisfies the question; redundant combined alternative is removed
    assert q14["expected_chunk_ids"] == [chunk_22]
    assert chunk_21 not in q14["expected_chunk_ids"]

    # Verify expected_evidence_sets structure
    assert "expected_evidence_sets" in q14
    ev_sets = q14["expected_evidence_sets"]
    assert ev_sets == [[chunk_22]]
    assert "independently answers" in q14["source_passage"]


def test_annotate_candidate_hits_complete_evidence_semantics():
    """
    Verify that candidate-level is_hit strictly adheres to complete-evidence-set semantics:
    - Partial retrieval of a multi-chunk evidence set does NOT mark individual candidates as hits.
    - Complete retrieval of a multi-chunk evidence set marks all member candidates as hits.
    - Single-chunk evidence marks retrieved chunk as hit.
    - Alternative evidence sets only mark chunks belonging to fully completed sets.
    - Unanswerable queries have is_hit=False for all candidates.
    - Candidates outside top_k cutoff are never marked as hits.
    - any(c['is_hit'] for c in retrieved[:top_k]) is strictly consistent with hit_at_4.
    """
    # Scenario 1: Partial multi-chunk retrieval
    # Required evidence: {c1, c2}. Retrieved in top 4: c1 only.
    cands_partial = [
        {"chunk_id": "c1", "content": "part 1"},
        {"chunk_id": "cx", "content": "irrelevant"},
        {"chunk_id": "cy", "content": "irrelevant"},
        {"chunk_id": "cz", "content": "irrelevant"},
    ]
    annotate_candidate_hits(
        cands_partial,
        expected_evidence_sets=[["c1", "c2"]],
        top_k=4,
        is_answerable=True,
    )
    _, hit_at_4, _, _, _ = compute_ranked_metrics(
        cands_partial,
        expected_evidence_sets=[["c1", "c2"]],
        top_k=4,
    )
    assert not hit_at_4
    assert not cands_partial[0]["is_hit"], "Partial evidence chunk c1 must NOT be marked is_hit"
    assert not any(c["is_hit"] for c in cands_partial)

    # Scenario 2: Complete multi-chunk retrieval
    # Required evidence: {c1, c2}. Both retrieved in top 4.
    cands_complete = [
        {"chunk_id": "c1", "content": "part 1"},
        {"chunk_id": "cx", "content": "irrelevant"},
        {"chunk_id": "c2", "content": "part 2"},
        {"chunk_id": "cy", "content": "irrelevant"},
    ]
    annotate_candidate_hits(
        cands_complete,
        expected_evidence_sets=[["c1", "c2"]],
        top_k=4,
        is_answerable=True,
    )
    _, hit_at_4, _, _, _ = compute_ranked_metrics(
        cands_complete,
        expected_evidence_sets=[["c1", "c2"]],
        top_k=4,
    )
    assert hit_at_4
    assert cands_complete[0]["is_hit"] is True
    assert cands_complete[1]["is_hit"] is False
    assert cands_complete[2]["is_hit"] is True
    assert cands_complete[3]["is_hit"] is False

    # Scenario 3: Alternative sets: {c1} complete, {c2, c3} incomplete
    cands_alt = [
        {"chunk_id": "cx", "content": "irrelevant"},
        {"chunk_id": "c2", "content": "part of incomplete set"},
        {"chunk_id": "c1", "content": "sufficient alone"},
        {"chunk_id": "cy", "content": "irrelevant"},
    ]
    annotate_candidate_hits(
        cands_alt,
        expected_evidence_sets=[["c1"], ["c2", "c3"]],
        top_k=4,
        is_answerable=True,
    )
    assert cands_alt[2]["is_hit"] is True, "c1 is part of completed set {c1}"
    assert cands_alt[1]["is_hit"] is False, "c2 is part of incomplete set {c2, c3} and must NOT be marked is_hit"
    assert cands_alt[0]["is_hit"] is False
    assert cands_alt[3]["is_hit"] is False

    # Scenario 4: Candidate outside top_k cutoff
    cands_cutoff = [
        {"chunk_id": "cx"}, {"chunk_id": "cy"}, {"chunk_id": "cz"}, {"chunk_id": "cw"},
        {"chunk_id": "c1"},
    ]
    annotate_candidate_hits(
        cands_cutoff,
        expected_evidence_sets=[["c1"]],
        top_k=4,
        is_answerable=True,
    )
    assert cands_cutoff[4]["is_hit"] is False
    assert cands_cutoff[4]["strategy_rank"] == 5

    # Scenario 5: Unanswerable query
    cands_unans = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    annotate_candidate_hits(
        cands_unans,
        expected_evidence_sets=[["c1"]],
        top_k=4,
        is_answerable=False,
    )
    assert not any(c["is_hit"] for c in cands_unans)


def test_single_document_report_scope_isolation():
    """
    Ensure that when generating a report for a single-document evaluation:
    - Unrelated document names (CV, Optimization) do not appear in the report.
    - Multi-document claims ('three distinct technical documents', 'Multi-Document Corpus Insights') do not appear.
    - Section 7 is titled 'Single-Document Corpus Insights & Architecture Findings'.
    - Table 2 labels properly reflect the single-doc evaluation scope:
      'Run 1: V1 Initial Baseline (1 Doc, 97 chunks)'
      'Run 2: V1 Single-Doc Evaluation (1 Doc, 97 chunks)'
      'Run 3: V2 Multi-Doc (3 Docs, 246 chunks)'
    """
    metadata = {
        "evaluation_scope": {
            "knowledge_base_id": "54335692-b9bd-4693-b074-5b536bdd51af",
            "knowledge_base_name": "Cloud Computing Notes",
            "document_id": "607a4e68-8135-4686-a60b-5970d65fbada",
            "document_filename": "Cloud_Computing_Notes_Module1_2.docx",
            "total_chunks": 97,
        }
    }
    eval_results = [
        {
            "id": "Q01",
            "question": "What is cloud computing?",
            "answerable": True,
            "expected_document_id": "607a4e68-8135-4686-a60b-5970d65fbada",
            "strategies": {
                "dense_only": {"hit_at_1": True, "hit_at_4": True, "first_hit_rank": 1, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": [{"chunk_id": "c1", "document_id": "607a4e68-8135-4686-a60b-5970d65fbada", "filename": "Cloud_Computing_Notes_Module1_2.docx", "is_hit": True}]},
                "sparse_only": {"hit_at_1": True, "hit_at_4": True, "first_hit_rank": 1, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": []},
                "hybrid_rrf": {"hit_at_1": True, "hit_at_4": True, "first_hit_rank": 1, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": []},
                "hybrid_rerank": {"hit_at_1": True, "hit_at_4": True, "first_hit_rank": 1, "reciprocal_rank": 1.0, "ndcg": 1.0, "retrieved": [{"rerank_score": 0.95}]},
            },
        },
        {
            "id": "Q13",
            "question": "Off topic question?",
            "answerable": False,
            "category": "off_topic",
            "strategies": {
                "dense_only": {"hit_at_1": False, "hit_at_4": False, "retrieved": []},
                "sparse_only": {"hit_at_1": False, "hit_at_4": False, "retrieved": []},
                "hybrid_rrf": {"hit_at_1": False, "hit_at_4": False, "retrieved": []},
                "hybrid_rerank": {"hit_at_1": False, "hit_at_4": False, "retrieved": [{"rerank_score": 0.05, "document_id": "607a4e68-8135-4686-a60b-5970d65fbada", "filename": "Cloud_Computing_Notes_Module1_2.docx"}]},
            },
        },
    ]
    summary = calculate_metrics(eval_results)
    report = generate_markdown_report(summary, eval_results, metadata)

    # Scope and document isolation checks
    assert "Computer_Vision_Notes_Module1_2.docx" not in report
    assert "Optimization_for_ML_Notes_Module1_2.docx" not in report
    assert "three distinct technical documents" not in report
    assert "Multi-Document Corpus Insights" not in report
    assert "Single-Document Corpus Insights & Architecture Findings" in report
    assert "Run 1: V1 Initial Baseline (1 Doc, 97 chunks)" in report
    assert "Run 2: V1 Single-Doc Evaluation (1 Doc, 97 chunks)" in report
    assert "Run 3: V2 Multi-Doc (3 Docs, 246 chunks)" in report


@pytest.mark.asyncio
async def test_document_id_scope_isolation_sql_execution():
    """Verify that passing document_id adds document filtering to SQL query and parameters."""
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.mappings.return_value = []
    mock_db.execute.return_value = mock_result

    target_kb = "54335692-b9bd-4693-b074-5b536bdd51af"
    target_doc = "607a4e68-8135-4686-a60b-5970d65fbada"

    with patch("scripts.eval_retrieval.embedding_service.generate_embedding", return_value=[0.05] * 2000):
        candidates, vec = await fetch_retrieval_candidates(
            db=mock_db,
            query_text="Sample query",
            knowledge_base_id=target_kb,
            document_id=target_doc,
        )

    calls = mock_db.execute.call_args_list
    sql_call = calls[-1]
    sql_text = str(sql_call[0][0])
    params = sql_call[0][1]

    assert params["scope_id"] == target_kb
    assert params["document_id"] == target_doc
    assert "d.id = :document_id" in sql_text
