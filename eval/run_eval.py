"""
Agentix RAG Evaluation Framework
Executes end-to-end evaluation of the Agentix RAG pipeline and agent graph
across retrieval, abstention, hallucination, answer quality, and latency metrics.

Runs datasets sequentially:
1. eval/data/retrieval_questions_v2.json
2. eval/data/retrieval_questions_v2_borderline.json

Outputs:
- eval/results/<dataset_name>_results.json
- eval/results/report.md
- Terminal summary table
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

# Setup root and backend directories in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load backend/.env into environment before importing settings
from dotenv import load_dotenv
env_file = BACKEND_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file, override=False)

# Ensure HF_TOKEN is in environment to suppress HuggingFace unauthenticated warnings
if not os.getenv("HF_TOKEN") and os.getenv("HUGGINGFACE_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

# Allow CLI or environment override for LLM_MODEL, defaulting to gemini-3.5-flash-lite for reliable throughput
if "LLM_MODEL" not in os.environ or os.environ["LLM_MODEL"] == "gemini-3.5-flash":
    os.environ["LLM_MODEL"] = "gemini-3.5-flash-lite"

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Monkey-patch ChatGoogleGenerativeAI to enforce socket timeout and retries,
# preventing indefinite HTTP hangs during API socket drops
_orig_chat_init = ChatGoogleGenerativeAI.__init__
def _safe_chat_init(self, *args, **kwargs):
    if "timeout" not in kwargs or kwargs["timeout"] is None:
        kwargs["timeout"] = 35.0
    if "request_timeout" not in kwargs:
        kwargs["request_timeout"] = 35.0
    if "max_retries" not in kwargs or kwargs["max_retries"] is None:
        kwargs["max_retries"] = 2
    return _orig_chat_init(self, *args, **kwargs)
ChatGoogleGenerativeAI.__init__ = _safe_chat_init

from app.config.settings import settings
from app.database.connection import AsyncSessionLocal
from ai.agents.graph import rag_graph
from ai.agents.reflection import UNVERIFIED_ANSWER_CAVEAT
from ai.agents.state import AgentState
from ai.services.embedding_service import embedding_service, EmbeddingTask
from ai.services.retrieval_service import build_sparse_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")

# Decline detection phrases
DECLINE_PHRASES = [
    "couldn't find this in your uploaded documents",
    "could not find this in your uploaded documents",
    "not found in your uploaded documents",
    "not mentioned in the documents",
    "not covered in the uploaded documents",
    "not found in the uploaded document",
    "cannot find this in your uploaded documents",
    "no information",
    "does not contain",
    "search the web instead",
]


def is_system_declined(final_response: str, chunks_surviving: int) -> bool:
    """
    Checks if the system declined to answer based on:
    1. Zero chunks passed threshold
    2. Final answer text explicitly states information is missing from documents
    """
    if chunks_surviving == 0:
        return True
    resp_lower = final_response.lower()
    return any(phrase in resp_lower for phrase in DECLINE_PHRASES)


async def check_db_chunks_exist(db: AsyncSession, expected_chunk_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Checks which expected chunk IDs exist in document_chunks table.
    Returns (present_ids, missing_ids).
    """
    if not expected_chunk_ids:
        return [], []
    stmt = text("SELECT id FROM document_chunks WHERE id = ANY(:ids)")
    chunk_uuids = []
    for cid in expected_chunk_ids:
        try:
            chunk_uuids.append(uuid.UUID(cid))
        except (ValueError, TypeError):
            pass
    res = await db.execute(stmt, {"ids": chunk_uuids})
    present_set = {str(row[0]) for row in res.fetchall()}
    present = [cid for cid in expected_chunk_ids if cid in present_set]
    missing = [cid for cid in expected_chunk_ids if cid not in present_set]
    return present, missing


async def execute_retrieval_pipeline(
    db: AsyncSession,
    query_text: str,
    knowledge_base_id: str,
    user_id: str,
    list_size: int = 30,
    rrf_k: int = 60,
    rerank_candidates: int = 20,
    threshold: float = 0.30,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Executes hybrid retrieval (Dense + Sparse FTS) with RRF and Cross-Encoder reranking.
    Captures:
    (a) After RRF (top 4)
    (b) After Rerank (final 4 after threshold 0.30)
    """
    t0 = time.perf_counter()

    # 1. Dense query vector
    query_vector = await embedding_service.generate_embedding(query_text, mode=EmbeddingTask.QUERY)
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # 2. Sparse query string
    sparse_query = build_sparse_query(query_text)

    # 3. Apply local HNSW ef_search parameter
    try:
        await db.execute(text(f"SET LOCAL hnsw.ef_search = {settings.HNSW_EF_SEARCH}"))
        if settings.HNSW_ITERATIVE_SCAN and settings.HNSW_ITERATIVE_SCAN.lower() != "off":
            await db.execute(text(f"SET LOCAL hnsw.iterative_scan = '{settings.HNSW_ITERATIVE_SCAN}'"))
    except Exception as e:
        logger.warning(f"Could not apply local HNSW settings: {e}")

    # 4. Construct hybrid SQL
    scope_filter = "d.knowledge_base_id = :kb_id AND d.user_id = :user_id AND d.status = 'ready'"
    params = {
        "query_vector": query_vector_str,
        "sparse_query": sparse_query,
        "kb_id": str(knowledge_base_id),
        "user_id": str(user_id),
        "list_size": list_size,
        "rrf_k": rrf_k,
        "rerank_candidates": rerank_candidates,
    }

    if sparse_query:
        sparse_cte = f"""
        sparse_inner AS (
            SELECT
                dc.id,
                ts_rank_cd(dc.fts_tokens, websearch_to_tsquery('english', :sparse_query)) as rank_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {scope_filter}
              AND dc.fts_tokens IS NOT NULL
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
          AND dc.embedding IS NOT NULL
        ORDER BY distance
        LIMIT :list_size
    ),
    dense AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY distance) as rank
        FROM dense_inner
    )"""

    hybrid_sql = f"""
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
    ORDER BY rrf_score DESC, chunk_id
    LIMIT :rerank_candidates;
    """

    res = await db.execute(text(hybrid_sql), params)

    candidates = []
    for row in res.mappings():
        candidates.append({
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "content": row["content"],
            "filename": row["filename"],
            "score": float(row["rrf_score"]),
            "rrf_score": float(row["rrf_score"]),
            "sparse_rank": int(row["sparse_rank"]) if row["sparse_rank"] is not None else None,
            "dense_rank": int(row["dense_rank"]) if row["dense_rank"] is not None else None,
            "dense_similarity": float(row["dense_similarity"]) if row["dense_similarity"] is not None else None,
            "source_type": row["source_type"],
        })

    # Stage (a): Top 4 candidates by RRF score before reranking
    stage_a_top4 = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "chunk_index": c["chunk_index"],
            "page_number": c["page_number"],
            "filename": c["filename"],
            "rrf_score": c["rrf_score"],
            "dense_similarity": c["dense_similarity"],
            "source_type": c["source_type"],
            "dense_rank": c["dense_rank"],
            "sparse_rank": c["sparse_rank"],
        }
        for c in candidates[:top_k]
    ]

    # Stage (b): Rerank pool of 20 with Cross-Encoder
    rerank_pool = [dict(c) for c in candidates]
    if rerank_pool:
        reranked_pool = await embedding_service.rerank_chunks(query_text, rerank_pool)
    else:
        reranked_pool = []

    # Filter by threshold 0.30
    surviving_pool = [
        c for c in reranked_pool
        if (c.get("rerank_score") or 0.0) >= threshold
    ]

    stage_b_final4 = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "chunk_index": c["chunk_index"],
            "page_number": c["page_number"],
            "filename": c["filename"],
            "rerank_score": c.get("rerank_score"),
            "rerank_logit": c.get("rerank_logit"),
            "rrf_score": c["rrf_score"],
            "dense_similarity": c["dense_similarity"],
            "source_type": c["source_type"],
            "content": c["content"],
        }
        for c in surviving_pool[:top_k]
    ]

    top1_rerank_score = reranked_pool[0].get("rerank_score") if reranked_pool else None
    top1_dense_similarity = candidates[0].get("dense_similarity") if candidates else None

    latency_sec = time.perf_counter() - t0

    return {
        "stage_a_chunks": stage_a_top4,
        "stage_b_chunks": stage_b_final4,
        "candidates_retrieved": len(candidates),
        "chunks_passing_threshold": len(stage_b_final4),
        "final_chunks_passing_threshold": len(stage_b_final4),
        "pool_chunks_passing_threshold": len(surviving_pool),
        "top1_rerank_score": top1_rerank_score,
        "top1_dense_similarity": top1_dense_similarity,
        "retrieval_latency_ms": latency_sec * 1000.0,
    }


async def run_agent_graph(
    db: AsyncSession,
    query_text: str,
    knowledge_base_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Executes the full compiled LangGraph workflow:
    Supervisor -> Vector Search -> Answer -> Reflection -> Self-Correction/Verification.
    """
    session_id = str(uuid.uuid4())
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": str(user_id),
        "knowledge_base_id": str(knowledge_base_id),
        "allow_web_search": False,
        "user_query": query_text,
        "image_data": None,
        "extracted_image_query": None,
        "intent": None,
        "chat_history": [],
        "retrieved_chunks": [],
        "formatted_context": "",
        "context_source": None,
        "tool_outputs": [],
        "final_response": "",
        "error": None,
        "retry_count": 0,
        "previous_queries": [],
        "verified": None,
        "reflection_verdict": None,
    }

    t0 = time.perf_counter()
    final_state = None
    for attempt in range(2):
        try:
            final_state = await asyncio.wait_for(
                rag_graph.ainvoke(
                    initial_state,
                    config={"configurable": {"db": db}},
                ),
                timeout=180.0,
            )
            break
        except (asyncio.TimeoutError, Exception) as e:
            if attempt == 0:
                logger.warning(f"Agent graph invoke attempt 1 timed out or encountered error ({e}), retrying once...")
                await asyncio.sleep(2.0)
            else:
                logger.error(f"Agent graph invoke failed permanently: {e}")
                final_state = {
                    "final_response": "I could not find this in your uploaded documents.",
                    "intent": "TIMEOUT_FALLBACK",
                    "retry_count": 0,
                    "verified": False,
                    "reflection_verdict": "TIMEOUT",
                }
    latency_sec = time.perf_counter() - t0

    final_response = final_state.get("final_response", "")
    intent = final_state.get("intent", "UNKNOWN")
    retry_count = final_state.get("retry_count", 0)
    verified = final_state.get("verified")
    reflection_verdict = final_state.get("reflection_verdict")

    is_unverified = bool(
        verified is False
        or UNVERIFIED_ANSWER_CAVEAT in final_response
    )

    return {
        "final_response": final_response,
        "intent": intent,
        "retry_count": retry_count,
        "verified": verified,
        "reflection_verdict": reflection_verdict,
        "is_unverified": is_unverified,
        "e2e_latency_ms": latency_sec * 1000.0,
    }


async def evaluate_faithfulness(
    client: genai.Client,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "gemini-3.5-flash-lite",
) -> Dict[str, Any]:
    """
    Evaluates faithfulness using Gemini LLM Judge:
    "Given ONLY these retrieved chunks, is every claim in the answer supported? Reply JSON {supported: true/false, unsupported_claims: [...]}".
    """
    if not retrieved_chunks:
        return {"supported": False, "unsupported_claims": ["No document chunks retrieved to support answer."]}

    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {idx+1} | Source: {c.get('filename', 'doc')} | Page {c.get('page_number', 1)}]\n{c.get('content', '')}"
        for idx, c in enumerate(retrieved_chunks)
    )

    prompt = f"""Given ONLY these retrieved chunks, is every claim in the answer supported? Reply JSON {{supported: true/false, unsupported_claims: [...]}}.

Retrieved Chunks:
{chunks_text}

Answer:
{answer}
"""
    for attempt in range(3):
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                ),
                timeout=30.0,
            )
            raw = res.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw).strip()
            data = json.loads(raw)
            return {
                "supported": bool(data.get("supported")),
                "unsupported_claims": data.get("unsupported_claims", []),
            }
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Faithfulness judge call failed after 3 attempts: {e}")
                return {"supported": False, "unsupported_claims": [f"Judge invocation failed: {e}"]}
            await asyncio.sleep(2 ** attempt)

    return {"supported": False, "unsupported_claims": ["Judge call timeout"]}


async def evaluate_correctness(
    client: genai.Client,
    question: str,
    answer: str,
    source_passage: str,
    model_name: str = "gemini-3.5-flash-lite",
) -> Dict[str, Any]:
    """
    Evaluates answer correctness using Gemini LLM Judge:
    Compares the answer to the source_passage, JSON {correct: true/false}.
    """
    if not source_passage:
        return {"correct": False, "error": "No source passage available"}

    prompt = f"""Given the reference source passage, is the answer factually correct according to the passage? Reply JSON {{correct: true/false}}.

Source Passage:
{source_passage}

Question:
{question}

Answer:
{answer}
"""
    for attempt in range(3):
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                ),
                timeout=30.0,
            )
            raw = res.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw).strip()
            data = json.loads(raw)
            return {
                "correct": bool(data.get("correct")),
            }
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Correctness judge call failed after 3 attempts: {e}")
                return {"correct": False, "error": f"Judge invocation failed: {e}"}
            await asyncio.sleep(2 ** attempt)

    return {"correct": False, "error": "Judge call timeout"}


def compute_stage_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int = 4,
) -> Dict[str, Any]:
    """
    Computes retrieval metrics for cutoff K=4 on a single question:
    - Hit Rate@1, Hit Rate@4
    - MRR@4
    - Binary nDCG@4
    - Precision@4, Recall@4
    - MAP@4
    """
    expected_set = set(expected_chunk_ids)
    if not expected_set:
        return {
            "hit_at_1": 0.0,
            "hit_at_4": 0.0,
            "mrr_at_4": 0.0,
            "ndcg_at_4": 0.0,
            "precision_at_4": 0.0,
            "recall_at_4": 0.0,
            "map_at_4": 0.0,
            "first_hit_rank": None,
        }

    cutoff_ids = retrieved_chunk_ids[:k]

    # Find earliest hit rank
    first_hit_rank = None
    for idx, cid in enumerate(cutoff_ids):
        if cid in expected_set:
            first_hit_rank = idx + 1
            break

    hit_at_1 = 1.0 if first_hit_rank == 1 else 0.0
    hit_at_4 = 1.0 if (first_hit_rank is not None and first_hit_rank <= k) else 0.0
    mrr_at_4 = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    ndcg_at_4 = (1.0 / math.log2(first_hit_rank + 1)) if first_hit_rank is not None else 0.0

    # Relevant retrieved in top-k
    relevant_retrieved = sum(1 for cid in cutoff_ids if cid in expected_set)
    precision_at_4 = relevant_retrieved / float(k)
    recall_at_4 = relevant_retrieved / float(len(expected_set))

    # AP@4
    cum_hits = 0
    prec_sum = 0.0
    for idx, cid in enumerate(cutoff_ids):
        if cid in expected_set:
            cum_hits += 1
            prec_sum += cum_hits / float(idx + 1)
    map_at_4 = (prec_sum / float(len(expected_set))) if expected_set else 0.0

    return {
        "hit_at_1": hit_at_1,
        "hit_at_4": hit_at_4,
        "mrr_at_4": mrr_at_4,
        "ndcg_at_4": ndcg_at_4,
        "precision_at_4": precision_at_4,
        "recall_at_4": recall_at_4,
        "map_at_4": map_at_4,
        "first_hit_rank": first_hit_rank,
    }


def percentile(values: List[float], p: float) -> float:
    """Computes p-th percentile of values (0 to 100)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1


async def evaluate_dataset(
    dataset_path: Path,
    output_dir: Path,
    client: genai.Client,
    judge_model: str,
    limit: Optional[int] = None,
    run_name: str = "",
) -> Dict[str, Any]:
    """
    Evaluates one full dataset sequentially.
    """
    dataset_name = dataset_path.stem
    suffix = f"_{run_name}" if run_name else ""
    checkpoint_file = output_dir / f"{dataset_name}{suffix}_checkpoint.json"

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING EVALUATION: {dataset_name}{suffix} ({dataset_path.name})")
    print("=" * 80)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    scope = meta.get("evaluation_scope", {})
    kb_id = scope.get("knowledge_base_id")
    user_id = scope.get("user_id")

    questions = data.get("questions", [])
    if limit is not None:
        questions = questions[:limit]
    print(f"📋 Scope: KB={kb_id} | User={user_id} | Total Questions={len(questions)}")

    question_results = []
    failed_questions = []

    answerable_count = 0
    unanswerable_count = 0

    checkpoint_results: Dict[str, Any] = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_results = json.load(f)
            print(f"🔄 Checkpoint detected: {len(checkpoint_results)} questions previously completed.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}")

    # Ensure reranker is warmed up
    embedding_service.warmup_reranker()

    async with AsyncSessionLocal() as db:
        # Pre-check chunk existence in DB
        all_expected_ids = []
        for q in questions:
            if q.get("answerable"):
                all_expected_ids.extend(q.get("expected_chunk_ids", []))
        
        present_cids, missing_cids = await check_db_chunks_exist(db, all_expected_ids)
        if missing_cids:
            print(f"⚠️  WARNING: {len(missing_cids)} expected chunk IDs NOT FOUND in DB:")
            for mcid in missing_cids:
                print(f"    - Missing: {mcid}")
        else:
            print(f"✅ Verified: All {len(set(all_expected_ids))} ground-truth chunk IDs exist in the database.")

        for idx, q in enumerate(questions, start=1):
            qid = q["id"]
            q_text = q["question"]
            is_ans = bool(q.get("answerable"))
            topic = q.get("topic", "")
            exp_doc_id = q.get("expected_document_id")
            exp_chunk_ids = q.get("expected_chunk_ids", [])
            source_passage = q.get("source_passage", "")
            unans_cat = q.get("unanswerable_category", "answerable" if is_ans else "unanswerable")

            if is_ans:
                answerable_count += 1
            else:
                unanswerable_count += 1

            cached = checkpoint_results.get(qid)
            is_valid_cache = False
            if cached:
                intent_ok = cached.get("agent_execution", {}).get("intent") not in ("TIMEOUT_FALLBACK", "UNKNOWN")
                resp_text = cached.get("agent_execution", {}).get("final_response", "")
                resp_ok = not resp_text.startswith("An error occurred while generating the answer") and not resp_text.startswith("I couldn't search your documents right now")
                judge_ok = True
                if is_ans:
                    f_verdict = cached.get("judge_verdicts", {}).get("faithfulness")
                    c_verdict = cached.get("judge_verdicts", {}).get("correctness")
                    if not f_verdict or not c_verdict:
                        judge_ok = False
                    elif "Judge invocation failed" in str(f_verdict.get("unsupported_claims", [])):
                        judge_ok = False
                    elif "error" in c_verdict or "Judge invocation failed" in str(c_verdict.get("error", "")):
                        judge_ok = False
                is_valid_cache = intent_ok and resp_ok and judge_ok

            if is_valid_cache:
                item_result = cached
                question_results.append(item_result)
                if item_result.get("is_failed"):
                    failed_questions.append({
                        "id": qid,
                        "question": q_text,
                        "category": unans_cat,
                        "answerable": is_ans,
                        "expected_chunk_ids": exp_chunk_ids,
                        "stage_a_ids": [c["chunk_id"] for c in item_result["retrieval"]["stage_a_after_rrf_top4"]],
                        "stage_b_ids": [c["chunk_id"] for c in item_result["retrieval"]["stage_b_after_rerank_final4"]],
                        "top1_rerank_score": item_result["retrieval"]["top1_rerank_score"],
                        "failure_reasons": item_result.get("failure_reasons", []),
                        "final_response": item_result["agent_execution"]["final_response"][:200],
                    })
                print(f"[{idx}/{len(questions)}] ({qid}) [RESTORED FROM CHECKPOINT]")
                continue

            print(f"\n[{idx}/{len(questions)}] ({qid}) [{unans_cat}] {q_text[:70]}...")

            # Flag if expected chunks are missing in DB
            missing_ground_truth = [cid for cid in exp_chunk_ids if cid in missing_cids]
            if missing_ground_truth:
                print(f"   ⚠️ Expected chunk ID missing from DB: {missing_ground_truth}")

            # 1. Real Retrieval Pipeline (Hybrid + RRF + Rerank + Threshold 0.30)
            retrieval_res = await execute_retrieval_pipeline(
                db=db,
                query_text=q_text,
                knowledge_base_id=kb_id,
                user_id=user_id,
                list_size=30,
                rrf_k=60,
                rerank_candidates=20,
                threshold=0.30,
                top_k=4,
            )

            stage_a_chunks = retrieval_res["stage_a_chunks"]
            stage_b_chunks = retrieval_res["stage_b_chunks"]
            stage_a_ids = [c["chunk_id"] for c in stage_a_chunks]
            stage_b_ids = [c["chunk_id"] for c in stage_b_chunks]

            stage_a_metrics = compute_stage_retrieval_metrics(stage_a_ids, exp_chunk_ids, k=4) if is_ans else None
            stage_b_metrics = compute_stage_retrieval_metrics(stage_b_ids, exp_chunk_ids, k=4) if is_ans else None

            # 2. Run Full Agent Graph
            graph_res = await run_agent_graph(
                db=db,
                query_text=q_text,
                knowledge_base_id=kb_id,
                user_id=user_id,
            )

            final_response = graph_res["final_response"]
            system_declined = is_system_declined(final_response, retrieval_res["chunks_passing_threshold"])

            # 3. Abstention & Hallucination checks
            is_correct_abstention = False
            is_false_abstention = False
            is_hallucination = False

            if not is_ans:
                if system_declined:
                    is_correct_abstention = True
                else:
                    is_hallucination = True
            else:
                if system_declined:
                    is_false_abstention = True

            # 4. LLM Judges (Answerable only)
            faithfulness_res = None
            correctness_res = None
            if is_ans:
                faithfulness_res = await evaluate_faithfulness(
                    client=client,
                    answer=final_response,
                    retrieved_chunks=stage_b_chunks,
                    model_name=judge_model,
                )
                correctness_res = await evaluate_correctness(
                    client=client,
                    question=q_text,
                    answer=final_response,
                    source_passage=source_passage,
                    model_name=judge_model,
                )

            # Failure tracking
            is_failed = False
            failure_reasons = []
            if is_ans:
                if not stage_b_metrics["hit_at_4"]:
                    is_failed = True
                    failure_reasons.append("Retrieval Miss (Hit@4=0)")
                if is_false_abstention:
                    is_failed = True
                    failure_reasons.append("False Abstention (wrongly declined)")
                if faithfulness_res and not faithfulness_res["supported"]:
                    is_failed = True
                    failure_reasons.append(f"Unfaithful: {faithfulness_res.get('unsupported_claims')}")
                if correctness_res and not correctness_res["correct"]:
                    is_failed = True
                    failure_reasons.append("Incorrect Answer vs Source Passage")
            else:
                if is_hallucination:
                    is_failed = True
                    failure_reasons.append("Hallucination on unanswerable query (failed to decline)")

            if is_failed:
                failed_questions.append({
                    "id": qid,
                    "question": q_text,
                    "category": unans_cat,
                    "answerable": is_ans,
                    "expected_chunk_ids": exp_chunk_ids,
                    "stage_a_ids": stage_a_ids,
                    "stage_b_ids": stage_b_ids,
                    "top1_rerank_score": retrieval_res["top1_rerank_score"],
                    "failure_reasons": failure_reasons,
                    "final_response": final_response[:200],
                })

            item_result = {
                "id": qid,
                "question": q_text,
                "answerable": is_ans,
                "topic": topic,
                "unanswerable_category": unans_cat,
                "expected_document_id": exp_doc_id,
                "expected_chunk_ids": exp_chunk_ids,
                "missing_ground_truth_in_db": missing_ground_truth,
                "retrieval": {
                    "stage_a_after_rrf_top4": stage_a_chunks,
                    "stage_b_after_rerank_final4": stage_b_chunks,
                    "candidates_retrieved": retrieval_res["candidates_retrieved"],
                    "chunks_passing_threshold": retrieval_res["chunks_passing_threshold"],
                    "final_chunks_passing_threshold": retrieval_res.get("final_chunks_passing_threshold", len(stage_b_chunks)),
                    "pool_chunks_passing_threshold": retrieval_res.get("pool_chunks_passing_threshold", retrieval_res["chunks_passing_threshold"]),
                    "top1_rerank_score": retrieval_res["top1_rerank_score"],
                    "top1_dense_similarity": retrieval_res["top1_dense_similarity"],
                    "stage_a_metrics": stage_a_metrics,
                    "stage_b_metrics": stage_b_metrics,
                },
                "agent_execution": {
                    "intent": graph_res["intent"],
                    "retry_count": graph_res["retry_count"],
                    "verified": graph_res["verified"],
                    "reflection_verdict": graph_res["reflection_verdict"],
                    "is_unverified": graph_res["is_unverified"],
                    "system_declined": system_declined,
                    "is_correct_abstention": is_correct_abstention,
                    "is_false_abstention": is_false_abstention,
                    "is_hallucination": is_hallucination,
                    "final_response": final_response,
                    "answer_word_count": len(final_response.split()),
                },
                "judge_verdicts": {
                    "faithfulness": faithfulness_res,
                    "correctness": correctness_res,
                },
                "latency": {
                    "retrieval_ms": retrieval_res["retrieval_latency_ms"],
                    "e2e_answer_ms": graph_res["e2e_latency_ms"],
                },
                "is_failed": is_failed,
                "failure_reasons": failure_reasons,
            }
            question_results.append(item_result)
            checkpoint_results[qid] = item_result
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint_results, cf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update checkpoint file: {e}")

            top1_s = f"{retrieval_res['top1_rerank_score']:.4f}" if retrieval_res["top1_rerank_score"] is not None else "N/A"
            print(f"   [Retrieval] RRF Hit@4: {stage_a_metrics['hit_at_4'] if is_ans else 'N/A'} | Rerank Hit@4: {stage_b_metrics['hit_at_4'] if is_ans else 'N/A'} | Top-1 Score: {top1_s}")
            print(f"   [Agent] Intent={graph_res['intent']} | Declined={system_declined} | Retries={graph_res['retry_count']} | Unverified={graph_res['is_unverified']}")
            if is_ans:
                faith_str = str(faithfulness_res.get('supported')) if faithfulness_res else 'N/A'
                corr_str = str(correctness_res.get('correct')) if correctness_res else 'N/A'
                print(f"   [Judges] Faithful={faith_str} | Correct={corr_str}")
            print(f"   [Latency] Retrieval: {retrieval_res['retrieval_latency_ms']:.1f}ms | E2E: {graph_res['e2e_latency_ms']:.1f}ms | Words: {len(final_response.split())}")

    # Aggregate Metrics Calculation
    ans_stage_a = [q["retrieval"]["stage_a_metrics"] for q in question_results if q["answerable"]]
    ans_stage_b = [q["retrieval"]["stage_b_metrics"] for q in question_results if q["answerable"]]

    def avg_m(metric_list: List[Dict[str, Any]], key: str) -> float:
        if not metric_list:
            return 0.0
        return sum(item[key] for item in metric_list) / len(metric_list)

    # Retrieval side-by-side
    retrieval_summary = {
        "stage_a_rrf": {
            "hit_at_1": avg_m(ans_stage_a, "hit_at_1"),
            "hit_at_4": avg_m(ans_stage_a, "hit_at_4"),
            "mrr_at_4": avg_m(ans_stage_a, "mrr_at_4"),
            "ndcg_at_4": avg_m(ans_stage_a, "ndcg_at_4"),
            "precision_at_4": avg_m(ans_stage_a, "precision_at_4"),
            "recall_at_4": avg_m(ans_stage_a, "recall_at_4"),
            "map_at_4": avg_m(ans_stage_a, "map_at_4"),
        },
        "stage_b_rerank": {
            "hit_at_1": avg_m(ans_stage_b, "hit_at_1"),
            "hit_at_4": avg_m(ans_stage_b, "hit_at_4"),
            "mrr_at_4": avg_m(ans_stage_b, "mrr_at_4"),
            "ndcg_at_4": avg_m(ans_stage_b, "ndcg_at_4"),
            "precision_at_4": avg_m(ans_stage_b, "precision_at_4"),
            "recall_at_4": avg_m(ans_stage_b, "recall_at_4"),
            "map_at_4": avg_m(ans_stage_b, "map_at_4"),
        },
    }

    # Abstention metrics
    unans_items = [q for q in question_results if not q["answerable"]]
    ans_items = [q for q in question_results if q["answerable"]]

    correct_abstentions = sum(1 for q in unans_items if q["agent_execution"]["is_correct_abstention"])
    false_abstentions = sum(1 for q in ans_items if q["agent_execution"]["is_false_abstention"])
    hallucinations = sum(1 for q in unans_items if q["agent_execution"]["is_hallucination"])

    correct_abstention_rate = (correct_abstentions / len(unans_items)) if unans_items else 1.0
    false_abstention_rate = (false_abstentions / len(ans_items)) if ans_items else 0.0
    hallucination_rate = (hallucinations / len(unans_items)) if unans_items else 0.0

    overall_abstention_accuracy = (
        (correct_abstentions + (len(ans_items) - false_abstentions)) / len(question_results)
    ) if question_results else 0.0

    # Faithfulness & Correctness
    faithful_count = sum(1 for q in ans_items if q["judge_verdicts"]["faithfulness"] and q["judge_verdicts"]["faithfulness"]["supported"])
    correct_count = sum(1 for q in ans_items if q["judge_verdicts"]["correctness"] and q["judge_verdicts"]["correctness"]["correct"])
    pct_faithful = (faithful_count / len(ans_items) * 100.0) if ans_items else 0.0
    pct_correct = (correct_count / len(ans_items) * 100.0) if ans_items else 0.0

    # Context Quality
    # Context Precision: fraction of the 4 final chunks that are relevant
    context_precisions = []
    for q in ans_items:
        exp_set = set(q["expected_chunk_ids"])
        final_chunks = q["retrieval"]["stage_b_after_rerank_final4"]
        relevant_in_final = sum(1 for c in final_chunks if c["chunk_id"] in exp_set)
        # Using cutoff 4 as standard context precision
        context_precisions.append(relevant_in_final / 4.0)
    avg_context_precision = (sum(context_precisions) / len(context_precisions)) if context_precisions else 0.0

    ans_top1_rerank = [q["retrieval"]["top1_rerank_score"] for q in ans_items if q["retrieval"]["top1_rerank_score"] is not None]
    unans_top1_rerank = [q["retrieval"]["top1_rerank_score"] for q in unans_items if q["retrieval"]["top1_rerank_score"] is not None]

    avg_ans_top1_rerank = (sum(ans_top1_rerank) / len(ans_top1_rerank)) if ans_top1_rerank else 0.0
    avg_unans_top1_rerank = (sum(unans_top1_rerank) / len(unans_top1_rerank)) if unans_top1_rerank else 0.0

    avg_ans_final_chunks = sum(q["retrieval"].get("final_chunks_passing_threshold", len(q["retrieval"]["stage_b_after_rerank_final4"])) for q in ans_items) / len(ans_items) if ans_items else 0.0
    avg_unans_final_chunks = sum(q["retrieval"].get("final_chunks_passing_threshold", len(q["retrieval"]["stage_b_after_rerank_final4"])) for q in unans_items) / len(unans_items) if unans_items else 0.0
    avg_ans_pool_chunks = sum(q["retrieval"].get("pool_chunks_passing_threshold", q["retrieval"]["chunks_passing_threshold"]) for q in ans_items) / len(ans_items) if ans_items else 0.0
    avg_unans_pool_chunks = sum(q["retrieval"].get("pool_chunks_passing_threshold", q["retrieval"]["chunks_passing_threshold"]) for q in unans_items) / len(unans_items) if unans_items else 0.0

    # Answer length
    avg_ans_words = sum(q["agent_execution"].get("answer_word_count", len(q["agent_execution"]["final_response"].split())) for q in ans_items) / len(ans_items) if ans_items else 0.0
    avg_all_words = sum(q["agent_execution"].get("answer_word_count", len(q["agent_execution"]["final_response"].split())) for q in question_results) / len(question_results) if question_results else 0.0

    # Category breakdown of chunks passing
    category_chunks_passing = {}
    for q in question_results:
        cat = q["unanswerable_category"]
        if cat not in category_chunks_passing:
            category_chunks_passing[cat] = []
        category_chunks_passing[cat].append(q["retrieval"].get("final_chunks_passing_threshold", len(q["retrieval"]["stage_b_after_rerank_final4"])))
    avg_chunks_by_cat = {cat: (sum(lst) / len(lst)) for cat, lst in category_chunks_passing.items()}

    # Performance / Latencies
    retrieval_latencies = [q["latency"]["retrieval_ms"] for q in question_results]
    e2e_latencies = [q["latency"]["e2e_answer_ms"] for q in question_results]

    retrieval_p50 = percentile(retrieval_latencies, 50)
    retrieval_p95 = percentile(retrieval_latencies, 95)
    e2e_p50 = percentile(e2e_latencies, 50)
    e2e_p95 = percentile(e2e_latencies, 95)

    avg_retries = sum(q["agent_execution"]["retry_count"] for q in question_results) / len(question_results)
    unverified_count = sum(1 for q in question_results if q["agent_execution"]["is_unverified"])
    pct_unverified = (unverified_count / len(question_results)) * 100.0

    summary_data = {
        "dataset_name": dataset_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(question_results),
        "answerable_questions": len(ans_items),
        "unanswerable_questions": len(unans_items),
        "retrieval": retrieval_summary,
        "abstention": {
            "correct_abstention_rate": correct_abstention_rate,
            "false_abstention_rate": false_abstention_rate,
            "overall_abstention_accuracy": overall_abstention_accuracy,
        },
        "hallucination_and_quality": {
            "hallucination_rate": hallucination_rate,
            "faithfulness_pct": pct_faithful,
            "answer_correctness_pct": pct_correct,
        },
        "context_quality": {
            "context_precision": avg_context_precision,
            "avg_top1_rerank_answerable": avg_ans_top1_rerank,
            "avg_top1_rerank_unanswerable": avg_unans_top1_rerank,
            "separation_gap": avg_ans_top1_rerank - avg_unans_top1_rerank,
            "avg_final_chunks_passing_answerable": avg_ans_final_chunks,
            "avg_final_chunks_passing_unanswerable": avg_unans_final_chunks,
            "avg_pool_chunks_passing_answerable": avg_ans_pool_chunks,
            "avg_pool_chunks_passing_unanswerable": avg_unans_pool_chunks,
            "avg_chunks_passing_answerable": avg_ans_final_chunks,
            "avg_chunks_passing_unanswerable": avg_unans_final_chunks,
            "avg_chunks_passing_by_category": avg_chunks_by_cat,
        },
        "performance": {
            "retrieval_latency_p50_ms": retrieval_p50,
            "retrieval_latency_p95_ms": retrieval_p95,
            "e2e_latency_p50_ms": e2e_p50,
            "e2e_latency_p95_ms": e2e_p95,
            "average_retries": avg_retries,
            "pct_answers_unverified": pct_unverified,
            "avg_answer_words_answerable": avg_ans_words,
            "avg_answer_words_all": avg_all_words,
            "average_tokens_per_answer": "N/A (LangGraph AgentState and llm_service do not aggregate token usage metadata)",
        },
        "failed_questions_count": len(failed_questions),
        "failed_questions": failed_questions,
        "question_results": question_results,
    }

    # Save results JSON
    output_json_path = output_dir / f"{dataset_name}{suffix}_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n💾 Saved detailed per-question results to: {output_json_path}")

    return summary_data


def generate_markdown_report(results_list: List[Dict[str, Any]], report_path: Path) -> None:
    """
    Generates a clean Markdown report comparing both datasets with all required tables.
    """
    lines = []
    lines.append("# Agentix RAG Backend Evaluation Report")
    lines.append("")
    lines.append(f"> **Generated at**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("> **Pipeline Configuration**: 30 vector candidates, RRF $k=60$, 20 rerank candidates, 4 final chunks, Cross-Encoder threshold >= 0.30.")
    lines.append("")

    for dataset_summary in results_list:
        dname = dataset_summary["dataset_name"]
        lines.append(f"## Dataset: `{dname}`")
        lines.append("")
        lines.append(f"- **Total Questions**: {dataset_summary['total_questions']} ({dataset_summary['answerable_questions']} answerable, {dataset_summary['unanswerable_questions']} unanswerable)")
        lines.append(f"- **Failed Questions**: {dataset_summary['failed_questions_count']}")
        lines.append("")

        # 1. Retrieval Metrics Table (Side by Side Stage a vs Stage b)
        lines.append("### 1. Retrieval Performance: Before vs. After Reranker (Cutoff K=4, Answerable Only)")
        lines.append("")
        lines.append("| Metric | Stage (a): Hybrid RRF Top-4 (Before Reranker) | Stage (b): Cross-Encoder Final-4 (After Reranker & Threshold) | Delta / Impact |")
        lines.append("| :--- | :---: | :---: | :---: |")

        r_a = dataset_summary["retrieval"]["stage_a_rrf"]
        r_b = dataset_summary["retrieval"]["stage_b_rerank"]

        metrics_map = [
            ("Hit Rate@1", "hit_at_1", True),
            ("Hit Rate@4", "hit_at_4", True),
            ("MRR@4", "mrr_at_4", False),
            ("Binary nDCG@4", "ndcg_at_4", False),
            ("Precision@4", "precision_at_4", False),
            ("Recall@4", "recall_at_4", False),
            ("MAP@4", "map_at_4", False),
        ]

        for label, key, is_pct in metrics_map:
            val_a = r_a[key]
            val_b = r_b[key]
            delta = val_b - val_a
            if is_pct:
                str_a = f"{val_a * 100.0:.1f}%"
                str_b = f"{val_b * 100.0:.1f}%"
                str_delta = f"{delta * 100.0:+.1f}%"
            else:
                str_a = f"{val_a:.4f}"
                str_b = f"{val_b:.4f}"
                str_delta = f"{delta:+.4f}"
            lines.append(f"| **{label}** | {str_a} | **{str_b}** | {str_delta} |")

        lines.append("")

        # 2. Abstention & Hallucination Table
        lines.append("### 2. Abstention, Hallucination & Answer Quality")
        lines.append("")
        lines.append("| Metric Category | Metric Name | Value | Evaluation Basis / Target |")
        lines.append("| :--- | :--- | :---: | :--- |")
        ab = dataset_summary["abstention"]
        hq = dataset_summary["hallucination_and_quality"]
        cq = dataset_summary["context_quality"]
        perf = dataset_summary["performance"]

        lines.append(f"| **Abstention** | Correct Abstention Rate | **{ab['correct_abstention_rate']*100.0:.1f}%** | Unanswerable queries where system declined (target: 100%) |")
        lines.append(f"| **Abstention** | False Abstention Rate | **{ab['false_abstention_rate']*100.0:.1f}%** | Answerable queries wrongly declined (target: 0%) |")
        lines.append(f"| **Abstention** | Overall Abstention Accuracy | **{ab['overall_abstention_accuracy']*100.0:.1f}%** | Correct routing across all {dataset_summary['total_questions']} queries |")
        lines.append(f"| **Hallucination** | Hallucination Rate | **{hq['hallucination_rate']*100.0:.1f}%** | Unanswerable queries answered confidently (target: 0%) |")
        lines.append(f"| **Answer Quality** | Faithfulness (LLM Judge) | **{hq['faithfulness_pct']:.1f}%** | Answer claims strictly supported by retrieved chunks |")
        lines.append(f"| **Answer Quality** | Answer Correctness (LLM Judge) | **{hq['answer_correctness_pct']:.1f}%** | Answer factually matches ground-truth source passage |")

        lines.append("")

        # 3. Context Quality Table
        lines.append("### 3. Context Quality & Relevance Separation")
        lines.append("")
        lines.append("| Metric Name | Value | Description |")
        lines.append("| :--- | :---: | :--- |")
        lines.append(f"| **Context Precision** | **{cq['context_precision']:.4f}** ({cq['context_precision']*100.0:.1f}%) | Average fraction of top 4 chunks that are relevant |")
        lines.append(f"| **Avg Top-1 Rerank (Answerable)** | **{cq['avg_top1_rerank_answerable']:.4f}** | Cross-encoder confidence on answerable queries |")
        lines.append(f"| **Avg Top-1 Rerank (Unanswerable)** | **{cq['avg_top1_rerank_unanswerable']:.4f}** | Cross-encoder score on unanswerable queries |")
        lines.append(f"| **Threshold Separation Margin** | **{cq['separation_gap']:+.4f}** | Separation gap around the 0.30 threshold |")
        lines.append(f"| **Avg Final Chunks Passing >= 0.30 (Answerable)** | **{cq['avg_final_chunks_passing_answerable']:.2f} / 4** | Final surviving document chunks passed to answer agent (max 4) |")
        lines.append(f"| **Avg Pool Chunks Passing >= 0.30 (Answerable)** | **{cq['avg_pool_chunks_passing_answerable']:.2f} / 20** | Surviving chunks from candidate pool before top-4 truncation |")
        lines.append(f"| **Avg Final Chunks Passing >= 0.30 (Unanswerable)** | **{cq['avg_final_chunks_passing_unanswerable']:.2f} / 4** | Final leakage rate of irrelevant chunks (target: 0.0) |")
        lines.append(f"| **Avg Pool Chunks Passing >= 0.30 (Unanswerable)** | **{cq['avg_pool_chunks_passing_unanswerable']:.2f} / 20** | Leakage in candidate pool before top-4 truncation (target: 0.0) |")

        lines.append("")

        # 4. Performance & Operational Metrics
        lines.append("### 4. System Performance & Agent Reliability")
        lines.append("")
        lines.append("| Operational Metric | Value | Details |")
        lines.append("| :--- | :---: | :--- |")
        lines.append(f"| **Retrieval Latency p50** | **{perf['retrieval_latency_p50_ms']:.1f} ms** | Hybrid search + cross-encoder rerank median latency |")
        lines.append(f"| **Retrieval Latency p95** | **{perf['retrieval_latency_p95_ms']:.1f} ms** | Hybrid search + cross-encoder rerank 95th percentile |")
        lines.append(f"| **End-to-End Latency p50** | **{perf['e2e_latency_p50_ms'] / 1000.0:.2f} s** | Full agent graph execution median latency |")
        lines.append(f"| **End-to-End Latency p95** | **{perf['e2e_latency_p95_ms'] / 1000.0:.2f} s** | Full agent graph execution 95th percentile |")
        lines.append(f"| **Average Retries Used** | **{perf['average_retries']:.2f}** | Self-correction reflection retry loop count |")
        lines.append(f"| **Answers Marked Unverified** | **{perf['pct_answers_unverified']:.1f}%** | Responses flagged with unverified caveat |")
        lines.append(f"| **Average Answer Length (Words, Answerable)** | **{perf.get('avg_answer_words_answerable', 0.0):.1f} words** | Average words per answer for answerable queries |")
        lines.append(f"| **Average Answer Length (Words, All)** | **{perf.get('avg_answer_words_all', 0.0):.1f} words** | Average words per response across all queries |")
        lines.append(f"| **Average Tokens Per Answer** | **{perf['average_tokens_per_answer']}** | |")

        lines.append("")

        # 5. Failed Questions Section
        lines.append("### 5. Detailed Failure Analysis")
        lines.append("")
        failed = dataset_summary["failed_questions"]
        if not failed:
            lines.append("🎉 **Zero failures! All questions satisfied retrieval, abstention, and answer quality criteria.**")
        else:
            lines.append(f"Found **{len(failed)}** failed question(s) requiring inspection:")
            lines.append("")
            lines.append("| ID | Category | Question | Expected Chunk | Retrieved Top-1 Chunk (Score) | Failure Reasons |")
            lines.append("| :---: | :--- | :--- | :--- | :--- | :--- |")
            for fq in failed:
                exp_c = fq["expected_chunk_ids"][0] if fq["expected_chunk_ids"] else "None (Unanswerable)"
                ret_c = fq["stage_b_ids"][0] if fq["stage_b_ids"] else "None (Declined/Filtered)"
                top1_s = f"{fq['top1_rerank_score']:.4f}" if fq["top1_rerank_score"] is not None else "N/A"
                reasons_str = "; ".join(fq["failure_reasons"])
                lines.append(f"| **{fq['id']}** | `{fq['category']}` | {fq['question']} | `{exp_c}` | `{ret_c}` (score: {top1_s}) | {reasons_str} |")

        lines.append("\n---\n")

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n📄 Saved comprehensive evaluation report to: {report_path}")


def print_terminal_summary(results_list: List[Dict[str, Any]]) -> None:
    """
    Prints a clean, concise summary table in the terminal.
    """
    print("\n" + "=" * 95)
    print("                      AGENTIX RAG BENCHMARK EVALUATION SUMMARY")
    print("=" * 95)

    headers = f"{'Metric':<38} | " + " | ".join(f"{res['dataset_name'][:25]:^25}" for res in results_list)
    print(headers)
    print("-" * 95)

    def row(label, key_path, fmt="{:.1f}%", multiplier=100.0):
        vals = []
        for res in results_list:
            curr = res
            for k in key_path:
                curr = curr[k]
            if isinstance(curr, (int, float)):
                v = curr * multiplier
                try:
                    vals.append(fmt.format(v))
                except Exception:
                    vals.append(str(v))
            else:
                vals.append(str(curr))
        print(f"{label:<38} | " + " | ".join(f"{v:^25}" for v in vals))

    print("--- RETRIEVAL (Cutoff K=4) ---")
    row("Stage (a) RRF Hit Rate@4", ["retrieval", "stage_a_rrf", "hit_at_4"])
    row("Stage (b) Rerank Hit Rate@4", ["retrieval", "stage_b_rerank", "hit_at_4"])
    row("Stage (a) RRF Hit Rate@1", ["retrieval", "stage_a_rrf", "hit_at_1"])
    row("Stage (b) Rerank Hit Rate@1", ["retrieval", "stage_b_rerank", "hit_at_1"])
    row("Stage (a) RRF MRR@4", ["retrieval", "stage_a_rrf", "mrr_at_4"], fmt="{:.4f}", multiplier=1.0)
    row("Stage (b) Rerank MRR@4", ["retrieval", "stage_b_rerank", "mrr_at_4"], fmt="{:.4f}", multiplier=1.0)
    row("Stage (b) Binary nDCG@4", ["retrieval", "stage_b_rerank", "ndcg_at_4"], fmt="{:.4f}", multiplier=1.0)
    row("Stage (b) Precision@4", ["retrieval", "stage_b_rerank", "precision_at_4"], fmt="{:.4f}", multiplier=1.0)
    row("Stage (b) Recall@4", ["retrieval", "stage_b_rerank", "recall_at_4"], fmt="{:.4f}", multiplier=1.0)
    row("Stage (b) MAP@4", ["retrieval", "stage_b_rerank", "map_at_4"], fmt="{:.4f}", multiplier=1.0)

    print("\n--- ABSTENTION & HALLUCINATION ---")
    row("Correct Abstention Rate (Unans)", ["abstention", "correct_abstention_rate"])
    row("False Abstention Rate (Ans)", ["abstention", "false_abstention_rate"])
    row("Overall Abstention Accuracy", ["abstention", "overall_abstention_accuracy"])
    row("Hallucination Rate (Unans)", ["hallucination_and_quality", "hallucination_rate"])

    print("\n--- ANSWER QUALITY (LLM JUDGE) ---")
    row("Faithfulness % (Answerable)", ["hallucination_and_quality", "faithfulness_pct"], multiplier=1.0)
    row("Answer Correctness % (Answerable)", ["hallucination_and_quality", "answer_correctness_pct"], multiplier=1.0)

    print("\n--- CONTEXT QUALITY ---")
    row("Context Precision", ["context_quality", "context_precision"], fmt="{:.4f}", multiplier=1.0)
    row("Avg Top-1 Rerank (Answerable)", ["context_quality", "avg_top1_rerank_answerable"], fmt="{:.4f}", multiplier=1.0)
    row("Avg Top-1 Rerank (Unanswerable)", ["context_quality", "avg_top1_rerank_unanswerable"], fmt="{:.4f}", multiplier=1.0)
    row("Avg Final Chunks Passing (Ans)", ["context_quality", "avg_final_chunks_passing_answerable"], fmt="{:.2f}", multiplier=1.0)
    row("Avg Pool Chunks Passing (Ans)", ["context_quality", "avg_pool_chunks_passing_answerable"], fmt="{:.2f}", multiplier=1.0)
    row("Avg Chunks Passing (Unans)", ["context_quality", "avg_final_chunks_passing_unanswerable"], fmt="{:.2f}", multiplier=1.0)

    print("\n--- PERFORMANCE & AGENT STATS ---")
    row("Retrieval Latency p50 (ms)", ["performance", "retrieval_latency_p50_ms"], fmt="{:.1f}", multiplier=1.0)
    row("Retrieval Latency p95 (ms)", ["performance", "retrieval_latency_p95_ms"], fmt="{:.1f}", multiplier=1.0)
    row("End-to-End Latency p50 (s)", ["performance", "e2e_latency_p50_ms"], fmt="{:.2f}", multiplier=0.001)
    row("End-to-End Latency p95 (s)", ["performance", "e2e_latency_p95_ms"], fmt="{:.2f}", multiplier=0.001)
    row("Average Retries Used", ["performance", "average_retries"], fmt="{:.2f}", multiplier=1.0)
    row("% Answers Unverified", ["performance", "pct_answers_unverified"], multiplier=1.0)
    row("Avg Words (Answerable)", ["performance", "avg_answer_words_answerable"], fmt="{:.1f}", multiplier=1.0)
    row("Failed Questions Count", ["failed_questions_count"], fmt="{:d}", multiplier=1.0)
    print("=" * 95)


async def main():
    parser = argparse.ArgumentParser(description="Agentix RAG Evaluation Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "eval/data/retrieval_questions_v2.json",
            "eval/data/retrieval_questions_v2_borderline.json",
        ],
        help="Path(s) to evaluation dataset JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory to save output results and report",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        help="Gemini model for faithfulness and correctness judges",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Identifier for run (e.g. run1, run2) for reporting and checkpoint isolation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions per dataset (for testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    results_list = []
    for dset_str in args.datasets:
        dset_path = Path(dset_str)
        if not dset_path.is_file():
            print(f"❌ Error: Dataset file not found: {dset_path}")
            sys.exit(1)

        summary = await evaluate_dataset(
            dataset_path=dset_path,
            output_dir=out_dir,
            client=client,
            judge_model=args.judge_model,
            limit=args.limit,
            run_name=args.run_name,
        )
        results_list.append(summary)

    # Generate Markdown Report
    suffix = f"_{args.run_name}" if args.run_name else ""
    report_file = out_dir / f"report{suffix}.md"
    generate_markdown_report(results_list, report_file)

    # Print Terminal Summary Table
    print_terminal_summary(results_list)


if __name__ == "__main__":
    asyncio.run(main())
