"""
Benchmark Scoped Vector Search (Issue #29)
===========================================

Investigates query plan selection, latency, candidate count, and recall@k for
scoped dense (pgvector HNSW) and sparse (Postgres FTS) queries under various
scope sizes and pgvector settings.

Run Instructions:
-----------------
1. Ensure a throwaway PostgreSQL instance with pgvector is running. For example:
     docker run -d --name pgvector_throwaway_bench -p 5433:5432 \
       -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=bench_db \
       pgvector/pgvector:pg16

2. Apply Alembic migrations to set up the production schema:
     DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5433/bench_db \
       alembic upgrade head

3. Run this benchmark:
     python backend/scripts/benchmark_scoped_vector_search.py \
       --db-url postgresql://postgres:postgres@localhost:5433/bench_db \
       --scale-chunks 200000

Options:
  --db-url TEXT          PostgreSQL connection string (or set BENCHMARK_DATABASE_URL). Must be local/throwaway.
  --scale-chunks INT     Total number of chunks to generate (default: 200,000).
  --skip-gen             Skip synthetic data generation if already populated.
  --output-json PATH     Optional path to save raw results to a JSON file.
"""

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor


# Production query templates matching retrieval_service.py exactly
DENSE_QUERY_OR = """
SELECT
    dc.id,
    dc.embedding <=> CAST(%(query_vector)s AS vector) AS distance
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
WHERE (d.knowledge_base_id = %(kb_id)s OR d.session_id = %(session_id)s)
  AND d.user_id = %(user_id)s
  AND d.status = 'ready'
  AND dc.embedding IS NOT NULL
ORDER BY distance
LIMIT %(list_size)s;
"""

DENSE_QUERY_SINGLE_KB = """
SELECT
    dc.id,
    dc.embedding <=> CAST(%(query_vector)s AS vector) AS distance
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
WHERE d.knowledge_base_id = %(kb_id)s
  AND d.user_id = %(user_id)s
  AND d.status = 'ready'
  AND dc.embedding IS NOT NULL
ORDER BY distance
LIMIT %(list_size)s;
"""

DENSE_QUERY_SINGLE_SESSION = """
SELECT
    dc.id,
    dc.embedding <=> CAST(%(query_vector)s AS vector) AS distance
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
WHERE d.session_id = %(session_id)s
  AND d.user_id = %(user_id)s
  AND d.status = 'ready'
  AND dc.embedding IS NOT NULL
ORDER BY distance
LIMIT %(list_size)s;
"""

EXACT_DOC_RESTRICTED_OR = """
WITH scope_docs AS (
    SELECT id
    FROM documents
    WHERE (knowledge_base_id = %(kb_id)s OR session_id = %(session_id)s)
      AND user_id = %(user_id)s
      AND status = 'ready'
)
SELECT
    dc.id,
    dc.embedding <=> CAST(%(query_vector)s AS vector) AS distance
FROM document_chunks dc
WHERE dc.document_id IN (SELECT id FROM scope_docs)
  AND dc.embedding IS NOT NULL
ORDER BY distance
LIMIT %(list_size)s;
"""

EXACT_DOC_RESTRICTED_SINGLE_KB = """
WITH scope_docs AS (
    SELECT id
    FROM documents
    WHERE knowledge_base_id = %(kb_id)s
      AND user_id = %(user_id)s
      AND status = 'ready'
)
SELECT
    dc.id,
    dc.embedding <=> CAST(%(query_vector)s AS vector) AS distance
FROM document_chunks dc
WHERE dc.document_id IN (SELECT id FROM scope_docs)
  AND dc.embedding IS NOT NULL
ORDER BY distance
LIMIT %(list_size)s;
"""

SPARSE_QUERY_OR = """
SELECT
    dc.id,
    ts_rank_cd(dc.fts_tokens, websearch_to_tsquery('english', %(sparse_query)s)) as rank_score
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
WHERE (d.knowledge_base_id = %(kb_id)s OR d.session_id = %(session_id)s)
  AND d.user_id = %(user_id)s
  AND d.status = 'ready'
  AND dc.fts_tokens IS NOT NULL
  AND dc.fts_tokens @@ websearch_to_tsquery('english', %(sparse_query)s)
ORDER BY rank_score DESC
LIMIT %(list_size)s;
"""


def extract_plan_summary(plan_node: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Recursively inspects the EXPLAIN plan tree to find scan types."""
    details = []

    def walk(node):
        ntype = node.get("Node Type", "")
        rel = node.get("Relation Name", "")
        idx = node.get("Index Name", "")
        if rel:
            desc = f"{ntype} on {rel}"
            if idx:
                desc += f" using {idx}"
            details.append(desc)
        elif ntype:
            details.append(ntype)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_node)

    all_str = " | ".join(details)
    if "idx_document_chunks_embedding_hnsw" in all_str:
        summary = "HNSW Index Scan"
    elif "ix_document_chunks_document_id" in all_str:
        summary = "BTree DocID Scan"
    elif "Seq Scan on document_chunks" in all_str:
        summary = "Seq Scan (chunks)"
    elif "idx_document_chunks_fts" in all_str:
        summary = "GIN FTS Scan"
    elif "Bitmap Heap Scan on document_chunks" in all_str:
        summary = "Bitmap Scan (chunks)"
    else:
        summary = details[0] if details else "Unknown"

    return summary, details


def generate_synthetic_data(conn: psycopg2.extensions.connection, total_chunks: int = 200000):
    print(f"\n--- Generating synthetic dataset ({total_chunks:,} chunks) ---")
    cur = conn.cursor()
    
    # Check existing count
    cur.execute("SELECT count(*) FROM document_chunks;")
    existing = cur.fetchone()[0]
    if existing >= total_chunks * 0.95:
        print(f"Database already contains {existing:,} chunks. Skipping generation.")
        cur.close()
        return

    cur.execute("TRUNCATE users CASCADE;")
    conn.commit()
    
    # Temporarily drop indexes for fast bulk loading
    print("Dropping HNSW and GIN indexes for fast bulk insert...")
    cur.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding_hnsw;")
    cur.execute("DROP INDEX IF EXISTS idx_document_chunks_fts;")
    conn.commit()

    dim = 2000
    num_users = 50
    user_ids = [uuid.uuid4() for _ in range(num_users)]
    
    # Insert users
    print(f"Creating {num_users} users...")
    user_rows = io.StringIO()
    for i, u in enumerate(user_ids):
        user_rows.write(f"{u}\tuser_{i}@example.com\tUser {i}\t\\N\t2026-11-08 00:00:00\t2026-11-08 00:00:00\n")
    user_rows.seek(0)
    cur.copy_expert("COPY users (id, email, full_name, avatar_url, created_at, updated_at) FROM STDIN WITH (FORMAT text)", user_rows)
    conn.commit()

    target_user_id = user_ids[0]

    # Target user KBs
    kb_small_id = uuid.uuid4()
    kb_medium_id = uuid.uuid4()
    kb_large_id = uuid.uuid4()
    
    # Target user sessions
    session_kb_id = uuid.uuid4()     # session attached to kb_medium
    session_only_id = uuid.uuid4()   # session with no KB
    
    cur.execute("""
        INSERT INTO knowledge_bases (id, user_id, name, description, created_at, updated_at)
        VALUES
            (%s, %s, 'Small KB (50)', 'Small KB', NOW(), NOW()),
            (%s, %s, 'Medium KB (500)', 'Medium KB', NOW(), NOW()),
            (%s, %s, 'Large KB (5000)', 'Large KB', NOW(), NOW());
    """, (str(kb_small_id), str(target_user_id),
          str(kb_medium_id), str(target_user_id),
          str(kb_large_id), str(target_user_id)))

    cur.execute("""
        INSERT INTO chat_sessions (id, user_id, title, knowledge_base_id, created_at, updated_at)
        VALUES
            (%s, %s, 'Session with KB Medium', %s, NOW(), NOW()),
            (%s, %s, 'Session Only', NULL, NOW(), NOW());
    """, (str(session_kb_id), str(target_user_id), str(kb_medium_id),
          str(session_only_id), str(target_user_id)))
    conn.commit()

    # Plan documents:
    # 1. kb_small: 1 doc, 50 chunks
    # 2. kb_medium: 2 docs, 250 chunks each = 500 chunks
    # 3. kb_large: 20 docs, 250 chunks each = 5,000 chunks
    # 4. session_kb upload: 1 doc, 50 chunks (session_id = session_kb_id, kb_id = NULL)
    # 5. session_only upload: 1 doc, 50 chunks (session_id = session_only_id, kb_id = NULL)
    # 6. target_user general docs: 5 docs, 200 chunks each = 1,000 chunks
    # Total target user chunks = 50 + 500 + 5000 + 50 + 50 + 1000 = 6,650 chunks across 30 docs.
    # Remaining chunks ~ 193,350 across ~970 docs distributed among 49 other users.

    docs_plan = []
    # Scope documents:
    docs_plan.append((target_user_id, kb_small_id, None, 50, "small_kb_doc1.pdf"))
    docs_plan.append((target_user_id, kb_medium_id, None, 250, "medium_kb_doc1.pdf"))
    docs_plan.append((target_user_id, kb_medium_id, None, 250, "medium_kb_doc2.pdf"))
    for i in range(20):
        docs_plan.append((target_user_id, kb_large_id, None, 250, f"large_kb_doc_{i+1}.pdf"))
    docs_plan.append((target_user_id, None, session_kb_id, 50, "session_upload_in_kb_chat.pdf"))
    docs_plan.append((target_user_id, None, session_only_id, 50, "session_upload_standalone.pdf"))
    for i in range(5):
        docs_plan.append((target_user_id, None, None, 200, f"target_user_extra_doc_{i+1}.pdf"))

    target_chunks = sum(p[3] for p in docs_plan)
    remaining_chunks = total_chunks - target_chunks
    chunks_per_doc = 200
    num_other_docs = max(1, remaining_chunks // chunks_per_doc)
    
    print(f"Target user: {len(docs_plan)} documents, {target_chunks:,} chunks.")
    print(f"Other users: {num_other_docs} documents, ~{remaining_chunks:,} chunks.")

    for i in range(num_other_docs):
        u = user_ids[1 + (i % (num_users - 1))]
        c_count = chunks_per_doc if i < num_other_docs - 1 else (remaining_chunks - (num_other_docs - 1) * chunks_per_doc)
        docs_plan.append((u, None, None, c_count, f"other_user_doc_{i+1}.pdf"))

    total_docs = len(docs_plan)
    print(f"Inserting {total_docs:,} documents...")
    doc_ids = [uuid.uuid4() for _ in range(total_docs)]
    
    doc_rows = io.StringIO()
    for d_id, (u_id, kb_id, sess_id, _, fname) in zip(doc_ids, docs_plan):
        kb_str = f"{kb_id}" if kb_id else "\\N"
        sess_str = f"{sess_id}" if sess_id else "\\N"
        doc_rows.write(f"{d_id}\t{u_id}\t{sess_str}\t{fname}\tpdf\t/fake/{fname}\t102400\tready\t\\N\t2026-11-08 00:00:00\t{kb_str}\t2026-11-08 00:00:00\t\\N\t0\n")
    doc_rows.seek(0)
    cur.copy_expert("COPY documents (id, user_id, session_id, filename, file_type, file_path, file_size, status, error_message, created_at, knowledge_base_id, updated_at, content_hash, ocr_warning_count) FROM STDIN WITH (FORMAT text)", doc_rows)
    conn.commit()

    # Now generate chunks in batches
    print(f"Generating and inserting {total_chunks:,} chunks (dim={dim}) in batches...")
    batch_size = 5000
    chunk_buffer = io.StringIO()
    inserted_chunks = 0
    t0 = time.time()

    topics = [
        "refund policy customer dispute invoice payment terms",
        "vector database hnsw iterative scan cosine similarity index",
        "authentication authorization jwt session token permissions",
        "data ingestion extraction parsing optical character recognition",
        "machine learning cross encoder reranker hybrid search fusion",
    ]

    for d_idx, (d_id, (u_id, kb_id, sess_id, c_count, fname)) in enumerate(zip(doc_ids, docs_plan)):
        # Generate cluster center for this document
        center = np.random.randn(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        
        # Perturbations for all chunks in this doc
        noise = np.random.randn(c_count, dim).astype(np.float32) * 0.05
        vecs = center + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        if u_id == target_user_id:
            topic = "refund policy customer dispute invoice payment terms"
        else:
            topic = topics[d_idx % len(topics)]
        for c_idx in range(c_count):
            c_uuid = uuid.uuid4()
            v_str = "[" + ",".join(f"{x:.5f}" for x in vecs[c_idx]) + "]"
            text_content = f"{fname} section {c_idx+1}: Detailed operational procedures regarding {topic} and compliance guidelines."
            chunk_buffer.write(f"{c_uuid}\t{d_id}\t{c_idx}\t{c_idx//10+1}\t{text_content}\t{{}}\t{v_str}\t2026-11-08 00:00:00\n")
            inserted_chunks += 1

            if inserted_chunks % batch_size == 0 or inserted_chunks == total_chunks:
                chunk_buffer.seek(0)
                cur.copy_expert("COPY document_chunks (id, document_id, chunk_index, page_number, content, metadata, embedding, created_at) FROM STDIN WITH (FORMAT text)", chunk_buffer)
                conn.commit()
                chunk_buffer = io.StringIO()
                rate = inserted_chunks / max(0.1, time.time() - t0)
                print(f"  Inserted {inserted_chunks:,}/{total_chunks:,} chunks ({rate:.1f} chunks/sec)...")

    # Recreate Indexes
    print("\nRebuilding GIN index on fts_tokens...")
    t_idx0 = time.time()
    cur.execute("CREATE INDEX idx_document_chunks_fts ON document_chunks USING gin (fts_tokens);")
    conn.commit()
    print(f"GIN index built in {time.time() - t_idx0:.1f}s.")

    print("Rebuilding HNSW index on embedding (m=16, ef_construction=64)...")
    t_hnsw0 = time.time()
    cur.execute("""
        CREATE INDEX idx_document_chunks_embedding_hnsw 
        ON document_chunks USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()
    print(f"HNSW index built in {time.time() - t_hnsw0:.1f}s.")

    print("Running ANALYZE on documents and document_chunks...")
    cur.execute("ANALYZE documents;")
    cur.execute("ANALYZE document_chunks;")
    conn.commit()
    cur.close()
    print("Synthetic data setup complete!")


def run_benchmark(conn: psycopg2.extensions.connection) -> Dict[str, Any]:
    print("\n=======================================================")
    print("           SCOPED VECTOR SEARCH BENCHMARK               ")
    print("=======================================================")
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Fetch test IDs
    cur.execute("SELECT id FROM users WHERE email = 'user_0@example.com';")
    target_user_id = str(cur.fetchone()["id"])
    
    cur.execute("SELECT id, name FROM knowledge_bases WHERE user_id = %s;", (target_user_id,))
    kbs = {r["name"]: str(r["id"]) for r in cur.fetchall()}
    kb_small_id = kbs["Small KB (50)"]
    kb_medium_id = kbs["Medium KB (500)"]
    kb_large_id = kbs["Large KB (5000)"]
    
    cur.execute("SELECT id, title FROM chat_sessions WHERE user_id = %s;", (target_user_id,))
    sessions = {r["title"]: str(r["id"]) for r in cur.fetchall()}
    session_kb_id = sessions["Session with KB Medium"]
    session_only_id = sessions["Session Only"]

    # Verify scope chunk counts
    def get_chunk_count(query: str, params: dict) -> int:
        cur.execute(f"SELECT count(dc.id) FROM document_chunks dc JOIN documents d ON dc.document_id = d.id WHERE {query};", params)
        return cur.fetchone()["count"]

    count_small = get_chunk_count("d.knowledge_base_id = %(kb_id)s AND d.user_id = %(user_id)s", {"kb_id": kb_small_id, "user_id": target_user_id})
    count_med = get_chunk_count("d.knowledge_base_id = %(kb_id)s AND d.user_id = %(user_id)s", {"kb_id": kb_medium_id, "user_id": target_user_id})
    count_large = get_chunk_count("d.knowledge_base_id = %(kb_id)s AND d.user_id = %(user_id)s", {"kb_id": kb_large_id, "user_id": target_user_id})
    count_or = get_chunk_count("(d.knowledge_base_id = %(kb_id)s OR d.session_id = %(session_id)s) AND d.user_id = %(user_id)s", {"kb_id": kb_medium_id, "session_id": session_kb_id, "user_id": target_user_id})
    count_sess = get_chunk_count("d.session_id = %(session_id)s AND d.user_id = %(user_id)s", {"session_id": session_only_id, "user_id": target_user_id})

    print(f"Verified Scope Sizes:")
    print(f"  - Small KB:          {count_small} chunks")
    print(f"  - Medium KB:         {count_med} chunks")
    print(f"  - Large KB:          {count_large} chunks")
    print(f"  - Medium KB + Sess:  {count_or} chunks (OR scope)")
    print(f"  - Session Only:      {count_sess} chunks")

    # Pick a sample query vector from the small KB document to ensure in-scope neighbors exist
    cur.execute("""
        SELECT dc.embedding 
        FROM document_chunks dc 
        JOIN documents d ON dc.document_id = d.id 
        WHERE d.knowledge_base_id = %s 
        LIMIT 1;
    """, (kb_small_id,))
    raw_vec = cur.fetchone()["embedding"]
    # Perturb slightly to make a realistic query
    arr = np.array([float(x) for x in raw_vec[1:-1].split(",")])
    query_vec_inscope = arr + np.random.randn(2000).astype(np.float32) * 0.02
    query_vec_inscope /= np.linalg.norm(query_vec_inscope)
    query_vec_str = "[" + ",".join(f"{x:.5f}" for x in query_vec_inscope) + "]"

    # Also pick an out-of-scope query vector (from another user's doc) to test post-filtering
    cur.execute("""
        SELECT dc.embedding 
        FROM document_chunks dc 
        JOIN documents d ON dc.document_id = d.id 
        WHERE d.user_id != %s 
        LIMIT 1;
    """, (target_user_id,))
    raw_other = cur.fetchone()["embedding"]
    arr_other = np.array([float(x) for x in raw_other[1:-1].split(",")])
    query_vec_other = arr_other + np.random.randn(2000).astype(np.float32) * 0.02
    query_vec_other /= np.linalg.norm(query_vec_other)
    query_vec_other_str = "[" + ",".join(f"{x:.5f}" for x in query_vec_other) + "]"

    scenarios = [
        ("Small KB (50 chunks)", DENSE_QUERY_SINGLE_KB, EXACT_DOC_RESTRICTED_SINGLE_KB, {"kb_id": kb_small_id, "user_id": target_user_id, "list_size": 30}),
        ("Medium KB (500 chunks)", DENSE_QUERY_SINGLE_KB, EXACT_DOC_RESTRICTED_SINGLE_KB, {"kb_id": kb_medium_id, "user_id": target_user_id, "list_size": 30}),
        ("Large KB (5000 chunks)", DENSE_QUERY_SINGLE_KB, EXACT_DOC_RESTRICTED_SINGLE_KB, {"kb_id": kb_large_id, "user_id": target_user_id, "list_size": 30}),
        ("KB + Session OR (550 chunks)", DENSE_QUERY_OR, EXACT_DOC_RESTRICTED_OR, {"kb_id": kb_medium_id, "session_id": session_kb_id, "user_id": target_user_id, "list_size": 30}),
        ("Session Only (50 chunks)", DENSE_QUERY_SINGLE_SESSION, EXACT_DOC_RESTRICTED_SINGLE_KB, {"session_id": session_only_id, "user_id": target_user_id, "list_size": 30}),
    ]

    variants = [
        {"name": "ef40, iter=off (pgvector default)", "ef": 40, "iter": "off", "exact_rewrite": False},
        {"name": "ef100, iter=off (Agentix settings)", "ef": 100, "iter": "off", "exact_rewrite": False},
        {"name": "ef200, iter=off", "ef": 200, "iter": "off", "exact_rewrite": False},
        {"name": "ef100, iter=relaxed_order", "ef": 100, "iter": "relaxed_order", "exact_rewrite": False},
        {"name": "ef100, iter=strict_order", "ef": 100, "iter": "strict_order", "exact_rewrite": False},
        {"name": "Exact Scoped Rewrite", "ef": 100, "iter": "off", "exact_rewrite": True},
    ]

    results = []

    for sc_name, sql_prod, sql_exact_rw, base_params in scenarios:
        print(f"\n=======================================================")
        print(f" Scenario: {sc_name}")
        print(f"=======================================================")

        # Run for both query types: In-scope centered vs Global out-of-scope centered
        for q_label, q_vec in [("In-Scope Query", query_vec_str), ("Out-of-Scope Global Query", query_vec_other_str)]:
            params = dict(base_params)
            params["query_vector"] = q_vec

            # Ground truth: exact search with index scans disabled
            cur.execute("SAVEPOINT sp_gt;")
            cur.execute("SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off;")
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql_prod}", params)
            gt_plan = cur.fetchall()[0]["QUERY PLAN"][0]
            gt_time = gt_plan["Execution Time"]
            cur.execute(sql_prod, params)
            gt_rows = cur.fetchall()
            cur.execute("ROLLBACK TO SAVEPOINT sp_gt;")

            gt_ids = [str(r["id"]) for r in gt_rows]
            gt_count = len(gt_ids)
            gt_top4 = set(gt_ids[:4])
            gt_top30 = set(gt_ids[:30])

            print(f"\n  Query Type: {q_label} | Ground Truth: {gt_count} rows in {gt_time:.2f} ms")

            for v in variants:
                if v["exact_rewrite"] and "Session Only" in sc_name:
                    continue  # skip rewrite on session only for brevity

                cur.execute("SAVEPOINT sp_v;")
                try:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {v['ef']};")
                    if v["iter"] != "off":
                        cur.execute(f"SET LOCAL hnsw.iterative_scan = '{v['iter']}';")
                    else:
                        cur.execute("SET LOCAL hnsw.iterative_scan = 'off';")
                except Exception as e:
                    print(f"    Warning setting params: {e}")

                query_to_run = sql_exact_rw if v["exact_rewrite"] else sql_prod

                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_to_run}", params)
                plan_json = cur.fetchall()[0]["QUERY PLAN"][0]
                exec_time = plan_json["Execution Time"]
                plan_summary, plan_details = extract_plan_summary(plan_json["Plan"])

                cur.execute(query_to_run, params)
                res_rows = cur.fetchall()
                cur.execute("ROLLBACK TO SAVEPOINT sp_v;")

                res_ids = [str(r["id"]) for r in res_rows]
                res_count = len(res_ids)
                
                # Compute recall
                recall_30 = len(set(res_ids).intersection(gt_top30)) / max(1, min(len(gt_top30), 30))
                recall_4 = len(set(res_ids[:4]).intersection(gt_top4)) / max(1, min(len(gt_top4), 4))

                record = {
                    "scenario": sc_name,
                    "query_type": q_label,
                    "variant": v["name"],
                    "plan": plan_summary,
                    "time_ms": exec_time,
                    "rows": res_count,
                    "requested": 30,
                    "recall_at_30": recall_30,
                    "recall_at_4": recall_4,
                    "plan_details": plan_details,
                }
                results.append(record)

                print(f"    [{v['name']:<32}] Plan: {plan_summary:<18} | Time: {exec_time:>6.2f} ms | Rows: {res_count:>2}/30 | Recall@30: {recall_30:.2f} | Recall@4: {recall_4:.2f}")

    # Sparse CTE benchmark
    print(f"\n=======================================================")
    print(f" Sparse Query Benchmark (GIN index check)")
    print(f"=======================================================")
    sparse_params = {
        "kb_id": kb_medium_id,
        "session_id": session_kb_id,
        "user_id": target_user_id,
        "sparse_query": "refund policy",
        "list_size": 30,
    }
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {SPARSE_QUERY_OR}", sparse_params)
    sp_plan = cur.fetchall()[0]["QUERY PLAN"][0]
    sp_time = sp_plan["Execution Time"]
    sp_summary, sp_details = extract_plan_summary(sp_plan["Plan"])
    cur.execute(SPARSE_QUERY_OR, sparse_params)
    sp_rows = cur.fetchall()
    print(f"  Sparse OR Query: Plan={sp_summary} | Time={sp_time:.2f} ms | Rows={len(sp_rows)}/30")
    print(f"  Details: {' -> '.join(sp_details[:3])}")

    sparse_result = {
        "scenario": "Sparse FTS Query",
        "plan": sp_summary,
        "time_ms": sp_time,
        "rows": len(sp_rows),
        "details": sp_details,
    }

    cur.close()
    return {"dense_results": results, "sparse_result": sparse_result}


def validate_throwaway_db_url(db_url: str) -> None:
    """Verifies that the target database URL points strictly to a local/throwaway host."""
    parsed = urllib.parse.urlparse(db_url)
    host = (parsed.hostname or "").lower()
    allowed_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
    if host not in allowed_hosts and not host.endswith(".local") and not host.startswith("throwaway"):
        raise ValueError(
            f"Security check failed: host '{host}' is not a permitted local/throwaway host. "
            "This benchmark generates synthetic test data and must strictly run against a throwaway database."
        )
    netloc = (parsed.netloc or "").lower()
    if "supabase" in netloc or "aws" in netloc or "pooler" in netloc:
        raise ValueError("Security check failed: benchmark must never run against shared, staging, or production databases.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark scoped dense and sparse vector search.")
    parser.add_argument(
        "--db-url",
        default=os.getenv("BENCHMARK_DATABASE_URL"),
        help="PostgreSQL connection URL for local/throwaway database (or set BENCHMARK_DATABASE_URL env var).",
    )
    parser.add_argument("--scale-chunks", type=int, default=200000, help="Total chunks to generate")
    parser.add_argument("--skip-gen", action="store_true", help="Skip data generation")
    parser.add_argument("--output-json", default=None, help="Optional path to save output JSON")
    args = parser.parse_args()

    if not args.db_url:
        parser.error(
            "No database URL provided. Please pass --db-url (e.g. postgresql://user:password@localhost:5433/bench_db) "
            "or set the BENCHMARK_DATABASE_URL environment variable. Must target a local/throwaway database."
        )

    validate_throwaway_db_url(args.db_url)

    print(f"Connecting to {args.db_url}...")
    conn = psycopg2.connect(args.db_url)

    if not args.skip_gen:
        generate_synthetic_data(conn, total_chunks=args.scale_chunks)

    res = run_benchmark(conn)
    conn.close()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved raw results to {args.output_json}")


if __name__ == "__main__":
    main()
