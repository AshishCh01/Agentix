# Agentic RAG — Follow-Up Audit (Post Six-Phase Remediation)

Date: 2026-08-30
Scope: `backend/` and `frontend/`, re-audited from scratch against current source (not against `docs/audit_report.md`'s line numbers). Every finding in `docs/audit_report.md` was independently re-checked by reading the current file directly and, where practical, by running the code (full backend `pytest`, `alembic heads`/`history`, frontend `lint`/`build`). No code was modified as part of this audit.

Six commits landed between the original audit and this one:
`853a678` gemini embedding model introduced → `0142c68` fixed web search reflection problem → `3f96c73` fixed critical backend security issues → `1aca3e7` fixed backend input limits → `c40639f` updated re-render performance (plus `37d8847` refactored by antigravity, immediately prior).

---

## Scoreboard

- **Original findings:** 30 (5 "fix first" + 5 AI/RAG + 7 backend/infra + 8 frontend + 5 test-suite, de-duplicated where the same issue was cross-listed)
- **FIXED:** 28
- **PARTIALLY FIXED:** 2
- **STILL PRESENT:** 0
- **NOT REPRODUCIBLE:** 0
- **New findings introduced by the six phases:** 1 (minor)

---

## Findings table

| # | Finding | File(s) | Verdict |
|---|---|---|---|
| 1 | Blocking embedding/rerank calls freeze the event loop for all users | `ai/services/embedding_service.py`, `ai/tools/vector_search.py`, `ai/services/retrieval_service.py`, `app/routes/upload.py` | **FIXED** |
| 2 | Chunk overlap computed then discarded on every boundary | `ai/services/chunking_service.py:64-71` | **FIXED** |
| 3 | Web-search fallback bypasses the hallucination/groundedness check | `ai/agents/graph.py` (`route_after_answer`, `web_search_node`) | **FIXED** |
| 4 | `alembic upgrade head` cannot bootstrap a working schema from empty | `alembic/versions/*` | **FIXED** |
| 5 | Streamed tokens can be written into the wrong chat session | `frontend/src/context/ChatContext.jsx`, `frontend/src/api/chatApi.js` | **FIXED** |
| 6 | Prompt injection surface: no untrusted-data framing around retrieved/web content | `ai/prompts/answer_prompt.py`, `ai/agents/answer.py` | **FIXED**† |
| 7 | Two independently-drifting hybrid-search implementations | `ai/tools/vector_search.py` vs `ai/services/retrieval_service.py` | **FIXED** |
| 8 | `.doc` uploads accepted as "supported" but always fail opaquely | `ai/services/parser_service.py:65-74` | **FIXED** |
| 9 | Chunk size not actually a hard cap (not truly recursive) | `ai/services/chunking_service.py` | **FIXED** |
| 10 | Hardcoded `model_used="gemini-3.5-flash"` in telemetry (3 drifting defaults) | `app/routes/chat.py`, `app/config/settings.py`, `ai/services/llm_service.py` | **FIXED** |
| 11 | Raw exception text returned to clients | `app/routes/chat.py`, `app/routes/search.py` | **FIXED** |
| 12 | No pagination on list endpoints | `app/routes/session.py`, `app/database/crud.py` | **FIXED** |
| 13 | `get_chat_session()` silently drops the ownership filter if `user_id` is falsy | `app/database/crud.py:116-136` | **FIXED** |
| 14 | Unbounded, per-process, non-shared `_last_sync_time` cache | `app/auth/dependencies.py` | **FIXED**† |
| 15 | No startup validation for `SUPABASE_JWT_SECRET` | `app/main.py` | **FIXED** |
| 16 | `Document`/`DocumentChunk` timestamps naive, use deprecated `datetime.utcnow()` | `app/models/document.py`, `app/models/document_chunks.py` | **FIXED** |
| 17 | No size limit on chat message/image payload; no max length on search query | `app/schemas/chat.py`, `app/schemas/search.py` | **FIXED** |
| 18 | Auth tokens stored in `localStorage` by default | `frontend/src/api/supabaseClient.js` | **PARTIALLY FIXED** |
| 19 | Advertised upload size limit doesn't match backend, not checked client-side | `frontend/src/components/Upload/UploadButton.jsx`, `frontend/src/hooks/useUpload.js` | **FIXED** |
| 20 | Unbounded base64 images retained in React state indefinitely | `frontend/src/components/Chat/ChatInput.jsx`, `frontend/src/context/ChatContext.jsx` | **FIXED** |
| 21 | `localStorage.clear()` on logout wipes unrelated app data | `frontend/src/context/AuthContext.jsx` | **FIXED** |
| 22 | Deferred session-clear on logout can race a fast re-login | `frontend/src/context/ChatContext.jsx` | **FIXED** |
| 23 | No memoization during token streaming (full list re-render per token) | `frontend/src/components/Chat/MessageItem.jsx`, `MessageList.jsx` | **FIXED** |
| 24 | `SourceBadge` uses array index as React key | `frontend/src/components/Chat/SourceBadge.jsx` | **FIXED** |
| 25 | Duplicated `fetchSessions` logic (two independently-drifting copies) | `frontend/src/context/ChatContext.jsx` | **FIXED** |
| 26 | Reflection max-retry cap has no test coverage at the graph level | `backend/tests/` | **FIXED** |
| 27 | Web-search-fallback-skips-reflection behavior has no test coverage | `backend/tests/` | **FIXED** |
| 28 | IDOR coverage partial (only `/search` and `/chat` tested) | `backend/tests/test_security.py` | **FIXED** |
| 29 | `test_db.py`/`test_gemini.py` always report PASSED regardless of outcome | `backend/tests/test_db.py`, `test_gemini.py` | **FIXED** |
| 30 | No test covers the chunk-overlap bug | `backend/tests/` | **FIXED** |

† Not a rubber-stamp "clean" — see Caveats below.

---

## Re-verification detail (fix-first five)

### 1. Blocking embedding/rerank calls — FIXED
`embedding_service.py` no longer wraps a local SentenceTransformer. `generate_embedding`/`generate_batch_embeddings` are `async def` and call `self.client.aio.models.embed_content(...)` (Gemini's native async client) — a real non-blocking await, not a sync call pretending to be async. `rerank_chunks` wraps the still-synchronous CrossEncoder `.predict()` in `asyncio.to_thread` (`embedding_service.py:109-112`), including the lazy model load on first access, so the one-time weight load also happens off the event loop. `vector_search.py` was collapsed to a thin wrapper that delegates to `retrieval_service.search_similar_chunks`, which itself only calls the async embedding method and the to-thread rerank. `upload.py`'s background ingestion path uses the same async batch-embedding call. No remaining sync CPU-bound call sits directly in an `async def` request handler.

### 2. Chunk overlap discarded — FIXED
`chunking_service.py:64-71`: the retained-overlap `while` loop now trims `current_doc` in place, and the trimmed list is kept (not overwritten with `[s]`) before continuing the loop — the overlap actually carries into the next chunk. `test_chunking.py::test_consecutive_chunks_share_overlap` asserts a real content-level overlap between consecutive chunks and passes.

### 3. Web-search fallback bypasses reflection — FIXED
`web_search_node` (`graph.py:145-162`) no longer overwrites `state["intent"]`. `route_after_answer` (`graph.py:57-76`) now branches on the supervisor's *original* classification, which survives the fallback — so a `RAG_QUERY` that fell back to `web_search` on an empty vector store still routes into `reflection`, while a query the supervisor classified as `WEB_SEARCH` directly still skips it, by design. `test_web_fallback_reflection.py` (8 tests) exercises both the unit-level routing functions and two full `rag_graph.ainvoke()` runs and confirms the reflection entry is/isn't present in `tool_outputs` as expected in each case.

### 4. Alembic cannot bootstrap from empty — FIXED
A new root migration, `ab5ddd3d013b_initial_schema_bootstrap.py`, creates the `vector` extension and every table (`users`, `chat_sessions`, `agent_logs`, `documents`, `messages`, `document_chunks`) from nothing. `e671bf5ba339`'s `down_revision` now points at it instead of `None`, and `f06fbea045e1` (previously two empty `pass` bodies) now issues real `CREATE INDEX IF NOT EXISTS` statements for the GIN/HNSW indexes. Verified live:
```
$ alembic heads
79afe98ca19a (head)
$ alembic history
f06fbea045e1 -> 79afe98ca19a (head), documents/document_chunks created_at timezone-aware
e671bf5ba339 -> f06fbea045e1, add vector and fts indexes
ab5ddd3d013b -> e671bf5ba339, baseline schema
<base> -> ab5ddd3d013b, initial schema bootstrap
```
Single linear head, no branching, and a genuinely empty database can now reach it via `alembic upgrade head` alone.

### 5. Streamed tokens land in wrong session — FIXED
`chatApi.js:streamMessage` now takes and forwards an `AbortController` `signal` to `fetch()`. `ChatContext.jsx` tracks `streamControllerRef`/`streamSessionIdRef`/`activeSessionIdRef`, gates every `onEvent`/`onError` state mutation behind `isStillActive()` (`activeSessionIdRef.current === targetSessionId`), and aborts the in-flight stream on session switch, session deletion, logout, and provider unmount. A `finally` block only clears stream state if nothing has superseded the current controller, so a stale response can't clobber a newer stream's bookkeeping either.

---

## Caveats on the two PARTIALLY FIXED items

- **#18 — Auth token storage.** `supabaseClient.js` now passes `storage: window.sessionStorage` explicitly instead of relying on Supabase's `localStorage` default. This is a real mitigation — the token is gone as soon as the tab/browser closes and isn't shared across tabs, shrinking both how long a stolen token stays valid and how far it can leak. It does **not** eliminate the underlying exposure class: `sessionStorage` is still a JS-readable Web Storage API, so a live XSS in the page (or a compromised transitive dependency) can still read and exfiltrate the token for the tab's lifetime, the same way it could with `localStorage`. A complete fix would require moving to an httpOnly cookie, which is a larger backend/CORS change. No XSS vector currently exists in the app (React's default escaping applies everywhere, no `dangerouslySetInnerHTML`), so this is a reasonable, deliberate trade-off rather than an oversight — but treat it as risk-reduced, not risk-eliminated.

- **#14 — Unbounded sync cache.** `_last_sync_time` is now an `OrderedDict` with LRU eviction capped at `_MAX_SYNC_CACHE_ENTRIES = 10_000` (`app/auth/dependencies.py`), closing the actual memory-leak failure scenario from the original finding, and `test_phase2_input_limits.py::test_last_sync_cache_evicts_oldest_beyond_max_size` verifies the eviction order. The cache is still per-process/non-shared across workers, which the fix's own comment explicitly acknowledges — in a multi-worker deployment, the "resync at most every 5 minutes" throttle is still unreliable across workers (each worker independently resyncs on first sight of a user). Closing this fully would require an external store (e.g. Redis), which is a separate infrastructure decision, not a code bug.

---

## New findings introduced by the six phases

### N1. Stale startup warmup logic with dead getattr-dispatch (Low severity)
`backend/app/main.py:25-40`

When embeddings moved from a local SentenceTransformer to the Gemini API (`853a678`), this lifespan block was not updated to match:

```python
print("[*] Pre-loading PyTorch and embedding model into RAM...")
try:
    embed_fn = (
        getattr(embedding_service, "embed_query", None)
        or getattr(embedding_service, "embed_text", None)
        or getattr(embedding_service, "generate_embedding", None)
        or getattr(embedding_service, "get_embeddings", None)
    )
    if callable(embed_fn):
        await embed_fn("warmup")
    print("[OK] Embedding model weights loaded into RAM successfully.")
except Exception as e:
    print(f"[WARN] Warning during embedding model pre-load: {e}")
```

Two of the four candidate method names (`embed_query`, `embed_text`, `get_embeddings`) have never existed on `EmbeddingService` — only `generate_embedding`, `generate_batch_embeddings`, and `rerank_chunks` do. This is the same speculative `hasattr`/`getattr` dispatch-to-a-nonexistent-method anti-pattern the original audit flagged and fixed in `vector_search.py` (finding #1), now reintroduced here. It is functionally harmless — the chain still resolves to the real `generate_embedding` — but:

- The log messages ("Pre-loading PyTorch... into RAM", "weights loaded into RAM successfully") are actively misleading: there is no local model being loaded into RAM for the embedding path anymore (that's now a cloud API call); only the CrossEncoder reranker uses local PyTorch weights, and it isn't warmed up here at all.
- Every process start/reload now makes a live, quota-consuming Gemini API call for a "warmup" that provides no actual benefit for an API-backed service (nothing to pre-load into RAM).

**Suggested fix:** replace the `getattr` chain with a direct `await embedding_service.generate_embedding("warmup")` call (or drop the warmup entirely, since there's no RAM-residency benefit to preserve), and correct the log strings.

No other new correctness, security, reliability, or performance regressions were found in the diffed surface area.

---

## Verification evidence

- **Backend test suite:** `pytest -v` from `backend/`, executed against a live Postgres/pgvector instance (not fully mocked) and the real Gemini client where exercised.
  ```
  46 passed, 1 skipped in 176.46s (0:02:56)
  ```
  The 1 skip is `test_gemini.py::test_gemini_connection`, skipped on a `429` quota response — by design (see that test's docstring), not a failure. All 5 new test files (`test_chunking.py`, `test_reflection_max_retry.py`, `test_web_fallback_reflection.py`, `test_phase1_security.py`, `test_phase2_input_limits.py`) were read in full and confirmed to exercise the actual fixed code paths (real DB round-trips via `AsyncSessionLocal`, full `rag_graph.ainvoke()` runs through `httpx.ASGITransport`, with only the outbound LLM/tool calls mocked) rather than asserting trivially true conditions.
- **Alembic:** `alembic heads` → single head `79afe98ca19a`; `alembic history` → clean linear chain from `<base>`, no branches. (See detail under finding #4 above.)
- **Frontend lint:** `npm run lint` (ESLint 10, `eslint.config.js` scoped to `**/*.{js,jsx}`, `dist/` ignored) → exit code 0, zero warnings/errors.
- **Frontend build:** `npm run build` (Vite 8) → exit code 0, `dist/` produced in 349ms. Only advisory output is a generic "chunk >500kB after minification" bundle-size warning, pre-existing and out of this audit's scope.
- Every FIXED/PARTIALLY FIXED verdict above reflects a direct read of the current file at the cited path plus, where applicable, a passing automated test — not the prior audit doc's own conclusions or the commit messages' claims.

---

## Recommended priority order for anything remaining

1. **N1** (stale `main.py` warmup logic) — trivial, ~10 minute fix; do opportunistically, no urgency.
2. **#18** (sessionStorage vs. httpOnly cookie) — revisit only if/when an XSS-capable surface is added to the app (e.g. rendering model output as HTML/markdown); no action needed under the current React-escaped-everywhere rendering model.
3. **#14** (cross-worker sync cache) — only matters once the deployment actually runs more than one worker process; move to Redis or equivalent at that point, not before.

Nothing found rises to "must fix before shipping." The six remediation phases closed every substantive finding from the original audit; the one new finding (N1) is cosmetic/wasteful, not broken.
