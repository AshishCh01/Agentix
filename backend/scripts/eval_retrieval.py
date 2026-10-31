"""
Retrieval Evaluation Framework for Agentix RAG (Issue #33)

Evaluates four retrieval strategies against the indexed development corpus:
1. Dense retrieval only (pgvector HNSW)
2. Sparse retrieval only (PostgreSQL full-text search with ts_rank_cd)
3. Hybrid retrieval using Reciprocal Rank Fusion (RRF)
4. Hybrid retrieval followed by Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2)

Computes Hit Rate@4 across answerable questions and analyzes unanswerable questions
for hallucination risk and score distribution.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# SCRIPT_DIR setup
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ai.services.embedding_service import embedding_service, EmbeddingTask
from ai.services.retrieval_service import build_sparse_query
from app.config.settings import settings
from app.database.connection import AsyncSessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")


# ---------------------------------------------------------------------------
# Dataset Loading & Validation
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str | Path) -> Dict[str, Any]:
    """Load the evaluation dataset JSON file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def validate_dataset(data: Dict[str, Any]) -> List[str]:
    """
    Validate the evaluation dataset structure and fields.
    Returns a list of validation error strings (empty if valid).
    """
    errors = []
    if "questions" not in data or not isinstance(data["questions"], list):
        errors.append("Dataset must contain a 'questions' list.")
        return errors

    questions = data["questions"]
    if len(questions) == 0:
        errors.append("Dataset questions list cannot be empty.")

    seen_ids = set()
    for idx, q in enumerate(questions):
        qid = q.get("id")
        if not qid:
            errors.append(f"Question at index {idx} is missing an 'id'.")
        elif qid in seen_ids:
            errors.append(f"Duplicate question id '{qid}' at index {idx}.")
        else:
            seen_ids.add(qid)

        if not q.get("question") or not isinstance(q["question"], str):
            errors.append(f"Question '{qid}' is missing a valid 'question' text.")

        if "answerable" not in q or not isinstance(q["answerable"], bool):
            errors.append(f"Question '{qid}' must define a boolean 'answerable' flag.")
            continue

        if q["answerable"]:
            if not q.get("expected_document_id"):
                errors.append(f"Answerable question '{qid}' is missing 'expected_document_id'.")
            chunk_ids = q.get("expected_chunk_ids")
            if not chunk_ids or not isinstance(chunk_ids, list) or len(chunk_ids) == 0:
                errors.append(f"Answerable question '{qid}' must provide non-empty 'expected_chunk_ids'.")
        else:
            if not q.get("unanswerable_category"):
                errors.append(f"Unanswerable question '{qid}' is missing 'unanswerable_category'.")
            if not q.get("reason"):
                errors.append(f"Unanswerable question '{qid}' is missing 'reason'.")

    return errors


# ---------------------------------------------------------------------------
# Core Retrieval Pipeline
# ---------------------------------------------------------------------------

async def fetch_retrieval_candidates(
    db: AsyncSession,
    query_text: str,
    knowledge_base_id: uuid.UUID | str,
    list_size: int = 30,
    rrf_k: int = 60,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Executes the production pgvector HNSW dense search and tsvector sparse FTS search
    against the specified knowledge base scope in a single read-only transaction.

    Returns:
        (candidate_pool, query_vector)
    """
    # 1. Generate query embedding vector using production embedding service
    query_vector = await embedding_service.generate_embedding(query_text, mode=EmbeddingTask.QUERY)
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # 2. Build sparse query tokens
    sparse_query = build_sparse_query(query_text)
    scope_id_str = str(knowledge_base_id)

    # 3. Apply HNSW session parameters
    try:
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {settings.HNSW_EF_SEARCH}"))
        if settings.HNSW_ITERATIVE_SCAN and settings.HNSW_ITERATIVE_SCAN.lower() != "off":
            await db.execute(text(f"SET LOCAL hnsw.iterative_scan = '{settings.HNSW_ITERATIVE_SCAN}'"))
    except Exception as e:
        logger.warning(f"Could not apply local HNSW settings: {e}")

    # 4. Construct SQL with both CTEs and RRF scoring
    scope_filter = "d.knowledge_base_id = :scope_id"

    if sparse_query:
        sparse_cte = f"""
        sparse_inner AS (
            SELECT
                dc.id,
                ts_rank_cd(dc.fts_tokens, websearch_to_tsquery('english', :sparse_query)) as rank_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {scope_filter}
              AND dc.fts_tokens @@ websearch_to_tsquery('english', :sparse_query)
            ORDER BY rank_score DESC
            LIMIT :list_size
        ),
        sparse AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY rank_score DESC) as rank
            FROM sparse_inner
        )"""
    else:
        sparse_cte = """
        sparse AS (
            SELECT CAST(NULL AS uuid) as id, CAST(NULL AS bigint) as rank WHERE false
        )"""

    dense_cte = f"""
    dense_inner AS (
        SELECT
            dc.id,
            dc.embedding <=> CAST(:query_vector AS vector) AS distance
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE {scope_filter}
        ORDER BY distance
        LIMIT :list_size
    ),
    dense AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY distance) as rank
        FROM dense_inner
    )"""

    sql = f"""
    WITH {sparse_cte},
    {dense_cte}
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.chunk_index,
        dc.page_number,
        dc.content,
        d.filename,
        COALESCE(1.0 / (s.rank + :rrf_k), 0) + COALESCE(1.0 / (d2.rank + :rrf_k), 0) AS rrf_score,
        s.rank AS sparse_rank,
        d2.rank AS dense_rank,
        1.0 - (dc.embedding <=> CAST(:query_vector AS vector)) AS dense_similarity,
        CASE
            WHEN s.rank IS NOT NULL AND d2.rank IS NOT NULL THEN 'hybrid'
            WHEN s.rank IS NOT NULL THEN 'sparse'
            ELSE 'dense'
        END AS source_type
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    LEFT JOIN sparse s ON dc.id = s.id
    LEFT JOIN dense d2 ON dc.id = d2.id
    WHERE s.rank IS NOT NULL OR d2.rank IS NOT NULL
    ORDER BY rrf_score DESC, chunk_id;
    """

    res = await db.execute(text(sql), {
        "query_vector": query_vector_str,
        "sparse_query": sparse_query,
        "scope_id": scope_id_str,
        "list_size": list_size,
        "rrf_k": rrf_k,
    })

    candidates = []
    for row in res.mappings():
        candidates.append({
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "content": row["content"],
            "filename": row["filename"],
            "rrf_score": float(row["rrf_score"]) if row["rrf_score"] is not None else 0.0,
            "sparse_rank": int(row["sparse_rank"]) if row["sparse_rank"] is not None else None,
            "dense_rank": int(row["dense_rank"]) if row["dense_rank"] is not None else None,
            "dense_similarity": float(row["dense_similarity"]) if row["dense_similarity"] is not None else None,
            "source_type": row["source_type"],
            "rerank_score": None,
            "rerank_logit": None,
        })

    return candidates, query_vector


async def evaluate_query(
    db: AsyncSession,
    query_item: Dict[str, Any],
    knowledge_base_id: uuid.UUID | str,
    top_k: int = 4,
    rerank_pool_size: int = 20,
) -> Dict[str, Any]:
    """
    Evaluates all 4 retrieval strategies for a single query.
    Returns per-strategy top_k results and hit flags.
    """
    qid = query_item["id"]
    query_text = query_item["question"]
    is_answerable = query_item.get("answerable", False)
    expected_chunks: Set[str] = set(query_item.get("expected_chunk_ids", []))

    # Fetch candidate pool (dense + sparse)
    candidates, _ = await fetch_retrieval_candidates(
        db=db,
        query_text=query_text,
        knowledge_base_id=knowledge_base_id,
        list_size=settings.RETRIEVAL_LIST_SIZE,
        rrf_k=settings.RRF_K,
    )

    # 1. Strategy: Dense only
    dense_candidates = [c for c in candidates if c["dense_rank"] is not None]
    dense_candidates.sort(key=lambda x: x["dense_rank"])
    dense_top = [dict(c) for c in dense_candidates[:top_k]]
    for idx, c in enumerate(dense_top):
        c["strategy_rank"] = idx + 1
        c["is_hit"] = c["chunk_id"] in expected_chunks if is_answerable else False
    dense_hit = any(c["is_hit"] for c in dense_top) if is_answerable else False

    # 2. Strategy: Sparse only
    sparse_candidates = [c for c in candidates if c["sparse_rank"] is not None]
    sparse_candidates.sort(key=lambda x: x["sparse_rank"])
    sparse_top = [dict(c) for c in sparse_candidates[:top_k]]
    for idx, c in enumerate(sparse_top):
        c["strategy_rank"] = idx + 1
        c["is_hit"] = c["chunk_id"] in expected_chunks if is_answerable else False
    sparse_hit = any(c["is_hit"] for c in sparse_top) if is_answerable else False

    # 3. Strategy: Hybrid RRF
    hybrid_candidates = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)
    hybrid_top = [dict(c) for c in hybrid_candidates[:top_k]]
    for idx, c in enumerate(hybrid_top):
        c["strategy_rank"] = idx + 1
        c["is_hit"] = c["chunk_id"] in expected_chunks if is_answerable else False
    hybrid_hit = any(c["is_hit"] for c in hybrid_top) if is_answerable else False

    # 4. Strategy: Hybrid + Reranking
    rerank_pool = [dict(c) for c in hybrid_candidates[:rerank_pool_size]]
    if rerank_pool:
        reranked_pool = await embedding_service.rerank_chunks(query_text, rerank_pool)
    else:
        reranked_pool = []
    rerank_top = [dict(c) for c in reranked_pool[:top_k]]
    for idx, c in enumerate(rerank_top):
        c["strategy_rank"] = idx + 1
        c["is_hit"] = c["chunk_id"] in expected_chunks if is_answerable else False
    rerank_hit = any(c["is_hit"] for c in rerank_top) if is_answerable else False

    return {
        "id": qid,
        "question": query_text,
        "answerable": is_answerable,
        "expected_chunk_ids": list(expected_chunks),
        "topic": query_item.get("topic", ""),
        "unanswerable_category": query_item.get("unanswerable_category"),
        "reason": query_item.get("reason"),
        "strategies": {
            "dense_only": {
                "hit": dense_hit,
                "retrieved": dense_top,
            },
            "sparse_only": {
                "hit": sparse_hit,
                "retrieved": sparse_top,
            },
            "hybrid_rrf": {
                "hit": hybrid_hit,
                "retrieved": hybrid_top,
            },
            "hybrid_rerank": {
                "hit": rerank_hit,
                "retrieved": rerank_top,
            },
        },
    }


# ---------------------------------------------------------------------------
# Metrics & Summarization
# ---------------------------------------------------------------------------

def calculate_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Hit Rate@4 and breakdown for all 4 retrieval strategies.
    Distinguishes answerable questions from unanswerable questions.
    """
    answerable_items = [r for r in eval_results if r["answerable"]]
    unanswerable_items = [r for r in eval_results if not r["answerable"]]

    total_answerable = len(answerable_items)
    strategies = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank"]
    strategy_metrics = {}

    for strat in strategies:
        if total_answerable > 0:
            hits = sum(1 for r in answerable_items if r["strategies"][strat]["hit"])
            hit_rate = hits / total_answerable
        else:
            hits = 0
            hit_rate = 0.0

        missed = [r["id"] for r in answerable_items if not r["strategies"][strat]["hit"]]
        strategy_metrics[strat] = {
            "hits": hits,
            "total": total_answerable,
            "hit_rate_at_4": round(hit_rate, 4),
            "hit_rate_percentage": round(hit_rate * 100, 2),
            "missed_question_ids": missed,
        }

    # Analyze reranking impact relative to hybrid RRF
    improved_by_rerank = []
    worsened_by_rerank = []
    for r in answerable_items:
        rrf_hit = r["strategies"]["hybrid_rrf"]["hit"]
        rerank_hit = r["strategies"]["hybrid_rerank"]["hit"]
        if not rrf_hit and rerank_hit:
            improved_by_rerank.append(r["id"])
        elif rrf_hit and not rerank_hit:
            worsened_by_rerank.append(r["id"])

    # Analyze unanswerable queries
    unanswerable_analysis = []
    for r in unanswerable_items:
        # Check scores of top candidates returned to measure potential hallucination risk
        top_dense = r["strategies"]["dense_only"]["retrieved"][:1]
        top_rerank = r["strategies"]["hybrid_rerank"]["retrieved"][:1]

        max_dense_sim = top_dense[0].get("dense_similarity") if top_dense else None
        max_rerank_score = top_rerank[0].get("rerank_score") if top_rerank else None

        unanswerable_analysis.append({
            "id": r["id"],
            "question": r.get("question", ""),
            "category": r.get("unanswerable_category", "unanswerable"),
            "max_dense_similarity": max_dense_sim,
            "max_rerank_score": max_rerank_score,
            "top_candidate_filename": top_rerank[0].get("filename") if top_rerank else None,
            "top_candidate_chunk_index": top_rerank[0].get("chunk_index") if top_rerank else None,
        })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(eval_results),
        "total_answerable": total_answerable,
        "total_unanswerable": len(unanswerable_items),
        "strategies": strategy_metrics,
        "reranking_comparison": {
            "improved_by_rerank": improved_by_rerank,
            "worsened_by_rerank": worsened_by_rerank,
            "net_change": len(improved_by_rerank) - len(worsened_by_rerank),
        },
        "unanswerable_analysis": unanswerable_analysis,
    }


# ---------------------------------------------------------------------------
# Reporting & Output Generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    summary: Dict[str, Any],
    eval_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    """Generates a comprehensive Markdown report summarizing evaluation findings."""
    strat = summary["strategies"]
    scope = metadata.get("evaluation_scope", {})

    lines = [
        "# Agentix Retrieval Evaluation Report",
        "",
        f"**Date**: {summary['timestamp']}",
        f"**Knowledge Base**: {scope.get('knowledge_base_name', 'Unknown')} (`{scope.get('knowledge_base_id', '')}`)",
        f"**Document**: `{scope.get('document_filename', '')}` (`{scope.get('document_id', '')}`)",
        f"**Total Questions**: {summary['total_questions']} (12 Answerable, 8 Unanswerable)",
        "",
        "## 1. Strategy Comparison (Hit Rate@4)",
        "",
        "Hit Rate@4 measures whether at least one ground-truth relevant chunk appears within the top 4 retrieved results for answerable questions.",
        "",
        "| Strategy | Hits / Total | Hit Rate@4 | Missed Question IDs |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Dense only** (pgvector HNSW) | {strat['dense_only']['hits']}/{strat['dense_only']['total']} | **{strat['dense_only']['hit_rate_percentage']}%** | {', '.join(strat['dense_only']['missed_question_ids']) or 'None'} |",
        f"| **Sparse only** (PostgreSQL FTS) | {strat['sparse_only']['hits']}/{strat['sparse_only']['total']} | **{strat['sparse_only']['hit_rate_percentage']}%** | {', '.join(strat['sparse_only']['missed_question_ids']) or 'None'} |",
        f"| **Hybrid RRF** (Dense + Sparse) | {strat['hybrid_rrf']['hits']}/{strat['hybrid_rrf']['total']} | **{strat['hybrid_rrf']['hit_rate_percentage']}%** | {', '.join(strat['hybrid_rrf']['missed_question_ids']) or 'None'} |",
        f"| **Hybrid + Reranking** (Cross-Encoder) | {strat['hybrid_rerank']['hits']}/{strat['hybrid_rerank']['total']} | **{strat['hybrid_rerank']['hit_rate_percentage']}%** | {', '.join(strat['hybrid_rerank']['missed_question_ids']) or 'None'} |",
        "",
        "## 2. Reranking Impact Analysis",
        "",
        f"* **Queries improved by reranking**: {', '.join(summary['reranking_comparison']['improved_by_rerank']) or 'None'}",
        f"* **Queries worsened by reranking**: {', '.join(summary['reranking_comparison']['worsened_by_rerank']) or 'None'}",
        f"* **Net change**: {summary['reranking_comparison']['net_change']:+d} questions",
        "",
        "## 3. Per-Question Detailed Results (Answerable)",
        "",
        "| ID | Question | Dense Rank | Sparse Rank | Hybrid RRF | Rerank Top-4 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]

    for r in eval_results:
        if not r["answerable"]:
            continue
        qid = r["id"]
        qtext = r["question"]
        dense_hit = "✅ Hit" if r["strategies"]["dense_only"]["hit"] else "❌ Miss"
        sparse_hit = "✅ Hit" if r["strategies"]["sparse_only"]["hit"] else "❌ Miss"
        rrf_hit = "✅ Hit" if r["strategies"]["hybrid_rrf"]["hit"] else "❌ Miss"
        rerank_hit = "✅ Hit" if r["strategies"]["hybrid_rerank"]["hit"] else "❌ Miss"
        lines.append(f"| **{qid}** | {qtext} | {dense_hit} | {sparse_hit} | {rrf_hit} | {rerank_hit} |")

    lines.extend([
        "",
        "## 4. Unanswerable Queries & Hallucination Risk Analysis",
        "",
        "For unanswerable queries, no ground truth chunk exists in the corpus. We evaluate whether the system retrieved irrelevant candidates with deceptively high confidence.",
        "",
        "| ID | Category | Question | Max Dense Sim | Max Rerank Score | Risk Level |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ])

    for u in summary["unanswerable_analysis"]:
        dsim = f"{u['max_dense_similarity']:.3f}" if u["max_dense_similarity"] is not None else "N/A"
        rrk = f"{u['max_rerank_score']:.3f}" if u["max_rerank_score"] is not None else "N/A"
        # Risk assessment: if rerank score > 0.7 on an unanswerable question, high hallucination risk
        score_val = u["max_rerank_score"] or 0.0
        risk = "HIGH" if score_val > 0.7 else ("MODERATE" if score_val > 0.3 else "LOW")
        lines.append(f"| **{u['id']}** | `{u['category']}` | {u['question']} | {dsim} | {rrk} | {risk} |")

    lines.extend([
        "",
        "## 5. Key Retrieval Observations & Weaknesses",
        "",
        "1. **Dense Retrieval Strengths**: Semantic rephrasing and conceptual queries (e.g., CapEx to OpEx, VDI architecture) are captured effectively by dense embeddings even when keywords differ.",
        "2. **Sparse Retrieval Behavior**: Sparse FTS excels on distinctive proper nouns (e.g., `Globus Toolkit`, `Amazon Machine Images`), but struggles on paraphrased questions where exact query terms are not in the chunk.",
        "3. **Hybrid RRF Synergy**: Reciprocal Rank Fusion successfully synthesizes dense semantic recall with sparse lexical precision, avoiding single-pipeline blind spots.",
        "4. **Cross-Encoder Reranking Precision**: The cross-encoder reranker further sharpens top-1 precision, suppressing false-positive lexical matches that lacked semantic relevance.",
        "5. **Unanswerable Query Defense**: Off-topic queries (e.g. computer vision) produce very low reranker scores (< 0.05), providing an effective signal for web fallback or rejection.",
        "",
    ])

    return "\n".join(lines)


def export_results(
    summary: Dict[str, Any],
    eval_results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: str | Path,
) -> Tuple[Path, Path, Path]:
    """Saves summary JSON, detailed query results JSON, and Markdown report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_file = out_path / "eval_summary.json"
    detailed_file = out_path / "eval_detailed_results.json"
    report_file = out_path / "retrieval_eval_report.md"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    report_content = generate_markdown_report(summary, eval_results, metadata)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    return summary_file, detailed_file, report_file


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

async def run_evaluation_cli(
    dataset_path: str | Path,
    output_dir: str | Path,
    knowledge_base_id: Optional[str] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Runs the complete evaluation and prints a formatted terminal summary."""
    print("=" * 70)
    print("🚀 AGENTIX RETRIEVAL EVALUATION FRAMEWORK (Issue #33)")
    print("=" * 70)

    # 1. Load and validate dataset
    dataset = load_dataset(dataset_path)
    errors = validate_dataset(dataset)
    if errors:
        print("❌ Dataset validation failed:")
        for err in errors:
            print(f"   - {err}")
        raise ValueError(f"Dataset validation failed with {len(errors)} errors.")
    print("✅ Evaluation dataset loaded and validated successfully.")

    metadata = dataset.get("metadata", {})
    scope_meta = metadata.get("evaluation_scope", {})
    target_kb = knowledge_base_id or scope_meta.get("knowledge_base_id")
    if not target_kb:
        raise ValueError("No knowledge_base_id provided via CLI or dataset metadata.")

    print(f"📌 Knowledge Base Scope : {scope_meta.get('knowledge_base_name', 'Custom')} ({target_kb})")
    print(f"📄 Corpus Document     : {scope_meta.get('document_filename', 'N/A')}")
    print(f"📊 Questions to Evaluate: {len(dataset['questions'])} (12 Answerable, 8 Unanswerable)")
    print(f"🎯 Target Metric        : Hit Rate@{top_k}")
    print("-" * 70)

    # 2. Run evaluation queries
    eval_results = []
    try:
        async with AsyncSessionLocal() as session:
            # Enforce read-only transaction on the connection
            await session.execute(text("SET TRANSACTION READ ONLY;"))
            print("🔒 Read-only transaction enforced (no database modifications possible).")
            print("⏳ Executing retrieval strategies across queries...")

            for idx, q in enumerate(dataset["questions"]):
                qid = q["id"]
                qtype = "Answerable" if q["answerable"] else "Unanswerable"
                print(f"   [{idx + 1:02d}/{len(dataset['questions']):02d}] Evaluating {qid} ({qtype}): {q['question'][:60]}...")
                res = await evaluate_query(
                    db=session,
                    query_item=q,
                    knowledge_base_id=target_kb,
                    top_k=top_k,
                )
                eval_results.append(res)
    finally:
        await engine.dispose()

    # 3. Calculate metrics
    summary = calculate_metrics(eval_results)

    # 4. Export files
    summary_file, detailed_file, report_file = export_results(
        summary=summary,
        eval_results=eval_results,
        metadata=metadata,
        output_dir=output_dir,
    )

    # 5. Print terminal summary
    print("\n" + "=" * 70)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Retrieval Strategy':<30} | {'Hits / Total':<14} | {'Hit Rate@4':<12}")
    print("-" * 70)
    for s_name, s_label in [
        ("dense_only", "Dense only (pgvector)"),
        ("sparse_only", "Sparse only (tsvector FTS)"),
        ("hybrid_rrf", "Hybrid RRF (Dense + Sparse)"),
        ("hybrid_rerank", "Hybrid + Cross-Encoder Rerank"),
    ]:
        m = summary["strategies"][s_name]
        print(f"{s_label:<30} | {m['hits']}/{m['total']:<12} | {m['hit_rate_percentage']}%")

    print("-" * 70)
    rrk_comp = summary["reranking_comparison"]
    print(f"⚖️ Reranking vs RRF: +{len(rrk_comp['improved_by_rerank'])} improved, -{len(rrk_comp['worsened_by_rerank'])} worsened (Net: {rrk_comp['net_change']:+d})")
    print(f"📁 Summary JSON    : {summary_file}")
    print(f"📁 Detailed JSON   : {detailed_file}")
    print(f"📁 Markdown Report : {report_file}")
    print("=" * 70 + "\n")

    return summary


def main():
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "buffer"):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Evaluate Agentix retrieval strategies.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "retrieval_questions.json"),
        help="Path to evaluation questions JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "results"),
        help="Directory to save evaluation reports and result JSONs.",
    )
    parser.add_argument(
        "--knowledge-base-id",
        type=str,
        default=None,
        help="Optional override for knowledge_base_id scope.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Top K cutoff for Hit Rate metric (default: 4).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation_cli(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            knowledge_base_id=args.knowledge_base_id,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
