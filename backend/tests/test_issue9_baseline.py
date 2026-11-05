"""
Unit tests for Issue #9 baseline evaluation helpers.
Uses synthetic fixtures with zero live database dependencies so CI runs reliably.
"""

import pytest
from scripts.eval_issue9_baseline import (
    compute_gap_analysis,
    extract_stage_top4,
    generate_baseline_markdown,
    get_git_commit_hash,
)


def test_extract_stage_top4_populates_fields_and_ranks():
    candidates = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "filename": "file1.docx",
            "chunk_index": 0,
            "dense_similarity": 0.85,
            "dense_rank": 1,
            "sparse_rank": 2,
            "rrf_score": 0.032,
            "rerank_score": None,
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",
            "filename": "file1.docx",
            "chunk_index": 1,
            "dense_similarity": 0.75,
            "dense_rank": 2,
            "sparse_rank": 1,
            "rrf_score": 0.030,
            "rerank_score": None,
        },
    ]
    rerank_scores_map = {"c1": 0.98, "c2": 0.45}

    result = extract_stage_top4(
        candidates=candidates,
        rerank_scores_map=rerank_scores_map,
        expected_chunk_ids=["c1"],
        is_answerable=True,
        top_k=2,
    )

    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"
    assert result[0]["final_rank"] == 1
    assert result[0]["is_hit"] is True
    assert result[0]["rerank_score"] == 0.98

    assert result[1]["chunk_id"] == "c2"
    assert result[1]["final_rank"] == 2
    assert result[1]["is_hit"] is False
    assert result[1]["rerank_score"] == 0.45


def test_compute_gap_analysis_clean_gap():
    eval_results = [
        {
            "id": "Q01",
            "question": "Ans 1",
            "category": "answerable",
            "answerable": True,
            "expected_document_id": "doc-a",
            "top1_scores": {
                "rerank_score": 0.95,
                "dense_similarity": 0.80,
                "top1_rerank_chunk_document": "DocA.docx",
                "top1_dense_chunk_document": "DocA.docx",
            },
        },
        {
            "id": "Q02",
            "question": "Ans 2",
            "category": "answerable",
            "answerable": True,
            "expected_document_id": "doc-b",
            "top1_scores": {
                "rerank_score": 0.85,
                "dense_similarity": 0.72,
                "top1_rerank_chunk_document": "DocB.docx",
                "top1_dense_chunk_document": "DocB.docx",
            },
        },
        {
            "id": "Q03",
            "question": "Unans 1",
            "category": "off_topic",
            "answerable": False,
            "top1_scores": {
                "rerank_score": 0.02,
                "dense_similarity": 0.55,
                "top1_rerank_chunk_document": "DocA.docx",
                "top1_dense_chunk_document": "DocA.docx",
            },
        },
    ]

    gap = compute_gap_analysis(eval_results)
    rrk = gap["rerank_score"]

    assert rrk["clean_gap"] is True
    assert rrk["lowest_answerable"]["question_id"] == "Q02"
    assert rrk["lowest_answerable"]["score"] == 0.85
    assert rrk["highest_unanswerable"]["question_id"] == "Q03"
    assert rrk["highest_unanswerable"]["score"] == 0.02
    assert round(rrk["gap_size"], 4) == 0.83
    assert rrk["overlapping_answerable_count"] == 0
    assert rrk["overlapping_unanswerable_count"] == 0


def test_compute_gap_analysis_overlap_detected():
    eval_results = [
        {
            "id": "Q01",
            "question": "Ans 1",
            "category": "answerable",
            "answerable": True,
            "expected_document_id": "doc-a",
            "top1_scores": {
                "rerank_score": 0.90,
                "dense_similarity": 0.65,
                "top1_rerank_chunk_document": "DocA.docx",
                "top1_dense_chunk_document": "DocA.docx",
            },
        },
        {
            "id": "Q02",
            "question": "Unans 1",
            "category": "same_subject_missing_detail",
            "answerable": False,
            "top1_scores": {
                "rerank_score": 0.05,
                "dense_similarity": 0.70,  # Higher than Q01 dense similarity -> overlap!
                "top1_rerank_chunk_document": "DocA.docx",
                "top1_dense_chunk_document": "DocA.docx",
            },
        },
    ]

    gap = compute_gap_analysis(eval_results)
    dense = gap["dense_similarity"]

    assert dense["clean_gap"] is False
    assert dense["gap_size"] < 0
    assert dense["overlapping_answerable_count"] == 1
    assert dense["overlapping_answerable"][0]["id"] == "Q01"
    assert dense["overlapping_unanswerable_count"] == 1
    assert dense["overlapping_unanswerable"][0]["id"] == "Q02"


def test_generate_baseline_markdown_contains_expected_tables():
    summary = {
        "strategies": {
            "dense_only": {"hit_rate_at_1_percentage": 55.56, "hits_at_1": 10, "hit_rate_percentage": 100.0, "hits_at_4": 18, "total": 18, "mrr_at_4": 0.7593, "ndcg_at_4": 0.8214},
            "sparse_only": {"hit_rate_at_1_percentage": 61.11, "hits_at_1": 11, "hit_rate_percentage": 83.33, "hits_at_4": 15, "total": 18, "mrr_at_4": 0.6944, "ndcg_at_4": 0.7291},
            "hybrid_rrf": {"hit_rate_at_1_percentage": 83.33, "hits_at_1": 15, "hit_rate_percentage": 94.44, "hits_at_4": 17, "total": 18, "mrr_at_4": 0.8796, "ndcg_at_4": 0.8962},
            "hybrid_rerank": {"hit_rate_at_1_percentage": 66.67, "hits_at_1": 12, "hit_rate_percentage": 100.0, "hits_at_4": 18, "total": 18, "mrr_at_4": 0.8056, "ndcg_at_4": 0.8552},
        }
    }
    eval_results = [
        {
            "id": "Q01",
            "question": "What is cloud computing?",
            "category": "answerable",
            "answerable": True,
            "top1_scores": {"rerank_score": 0.9912, "dense_similarity": 0.8451, "top1_rerank_chunk_document": "Cloud_Notes.docx"},
            "app_behavior": {"would_web_search_fire": False, "would_irrelevant_chunks_pass": False},
        }
    ]
    gap_analysis = {
        "rerank_score": {
            "lowest_answerable": {"question_id": "Q01", "question_text": "...", "score": 0.9912, "filename": "Cloud_Notes.docx"},
            "highest_unanswerable": {"question_id": "Q19", "question_text": "...", "category": "off_topic", "score": 0.0001},
            "clean_gap": True,
            "gap_size": 0.9911,
            "per_document_answerable_min": {},
            "unanswerable_by_category": {},
        },
        "dense_similarity": {
            "lowest_answerable": {"question_id": "Q01", "question_text": "...", "score": 0.8451, "filename": "Cloud_Notes.docx"},
            "highest_unanswerable": {"question_id": "Q19", "question_text": "...", "category": "off_topic", "score": 0.6123},
            "clean_gap": True,
            "gap_size": 0.2328,
            "overlapping_answerable_count": 0,
            "overlapping_unanswerable_count": 0,
            "overlapping_answerable": [],
            "overlapping_unanswerable": [],
            "per_document_answerable_min": {},
            "unanswerable_by_category": {},
        },
    }
    settings_meta = {
        "timestamp": "2026-11-05T00:00:00Z",
        "git_commit_hash": "abcdef123",
        "dataset_file": "retrieval_questions_v2.json",
        "dataset_version": "2.0",
        "knowledge_base_name": "Cloud Computing Notes",
        "knowledge_base_id": "kb-123",
        "total_documents": 3,
        "top_k": 4,
        "embedding_model": "gemini-embedding-2",
        "embedding_dimensions": 2000,
        "rerank_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rrf_k": 60,
        "retrieval_list_size": 30,
        "rerank_candidates": 20,
        "hnsw_ef_search": 100,
        "hnsw_iterative_scan": "relaxed_order",
    }

    md = generate_baseline_markdown(summary, eval_results, gap_analysis, settings_meta)
    assert "# Agentix Baseline Relevance & Routing Evaluation" in md
    assert "PR #50 Match?" in md
    assert "Key Table for Issue #9" in md
    assert "Gap Analysis" in md
    assert "Recommendations for Phase 2" in md


def test_safety_check_read_only_enforced():
    """Confirms by code inspection that eval_issue9_baseline enforces read-only transactions and no writes."""
    from pathlib import Path
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "eval_issue9_baseline.py"
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SET TRANSACTION READ ONLY;" in content
    # Confirm no mutating SQL operations are emitted
    for forbidden in ["INSERT INTO", "UPDATE ", "DELETE FROM", "DROP TABLE", "ALTER TABLE"]:
        assert forbidden not in content, f"Forbidden mutating SQL keyword found: {forbidden}"
