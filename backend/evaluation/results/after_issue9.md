# Agentix Relevance Threshold & Routing Verification (Issue #9 - Phase 2)

> **Phase 2 Fix Verification**: This report validates the relevance threshold implementation (`MIN_RELEVANCE_RERANK=0.3`, `MIN_RELEVANCE_DENSE=0.68`) and decoupled web-search routing against the Phase 1 baseline.

## 1. Executive Summary & Before / After Comparison

| Evaluation Metric | Phase 1 Baseline (Before Fix) | Phase 2 Verification (After Fix) | Delta / Status |
| :--- | :---: | :---: | :---: |
| **Answerable Hit Rate@4** (18 queries) | 100.0% (18/18) | **100.0% (18/18)** | ✅ Zero Regression |
| **Unanswerable Chunks Passing to LLM** | 100.0% (12/12 leak) | **0.0% (0/12)** | ✅ **100% Fixed (0 leaked)** |
| **Unanswerable 'Not Covered' Reply** | 0.0% (0/12) | **100.0% (12/12)** | ✅ **100% Honest Fallback** |
| **Automatic Web Search on Off-Topic** | Did not fire (bogus chunks) | **Did not fire (opt-in offer)** | ✅ Decoupled / Opt-in |
| **Relevance Threshold (`MIN_RELEVANCE_RERANK`)** | None (All 4 chunks kept) | **0.30** | ✅ Enforced on all chunks |

## 2. Strategy Accuracy Verification (V2 Dataset - 18 Answerable Queries)

| Strategy | Hit Rate@1 | Hit Rate@4 | MRR@4 | Binary nDCG@4 | Status vs Phase 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense only (pgvector HNSW)** | 55.6% | **100.0%** | **0.7593** | **0.8214** | ✅ Matches Baseline |
| **Sparse only (PostgreSQL FTS)** | 61.1% | **83.3%** | **0.6944** | **0.7291** | ✅ Matches Baseline |
| **Hybrid RRF (Dense + Sparse)** | 83.3% | **94.4%** | **0.8796** | **0.8962** | ✅ Matches Baseline |
| **Hybrid + Reranking (Thresholded)** | 66.7% | **100.0%** | **0.8056** | **0.8552** | ✅ Matches Baseline |

## 3. V2 Dataset: Per-Question Thresholding & Routing Status

| ID | Category | Top-1 Rerank | Chunks Kept | Route Decision | Irrelevant Passed? | Web Search Fires? | Response Summary |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q01** | `answerable` | **0.9997** | 2 | `answer` | ❌ No | ❌ No | [Answer synthesized from 2 relevant document chunk... |
| **Q02** | `answerable` | **0.9964** | 2 | `answer` | ❌ No | ❌ No | [Answer synthesized from 2 relevant document chunk... |
| **Q03** | `answerable` | **0.9989** | 3 | `answer` | ❌ No | ❌ No | [Answer synthesized from 3 relevant document chunk... |
| **Q04** | `answerable` | **0.9900** | 3 | `answer` | ❌ No | ❌ No | [Answer synthesized from 3 relevant document chunk... |
| **Q05** | `answerable` | **0.9998** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q06** | `answerable` | **0.9986** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q07** | `answerable` | **0.9984** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q08** | `answerable` | **0.9938** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q09** | `answerable` | **0.9997** | 1 | `answer` | ❌ No | ❌ No | [Answer synthesized from 1 relevant document chunk... |
| **Q10** | `answerable` | **0.9987** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q11** | `answerable` | **0.9878** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q12** | `answerable` | **0.8865** | 1 | `answer` | ❌ No | ❌ No | [Answer synthesized from 1 relevant document chunk... |
| **Q13** | `answerable` | **0.9904** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q14** | `answerable` | **0.9302** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q15** | `answerable` | **0.9977** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q16** | `answerable` | **0.9999** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q17** | `answerable` | **0.9968** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q18** | `answerable` | **0.9992** | 4 | `answer` | ❌ No | ❌ No | [Answer synthesized from 4 relevant document chunk... |
| **Q19** | `off_topic` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q20** | `off_topic` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q21** | `off_topic` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q22** | `off_topic` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q23** | `same_subject_missing_detail` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q24** | `same_subject_missing_detail` | **0.0001** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q25** | `same_subject_missing_detail` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q26** | `same_subject_missing_detail` | **0.0060** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q27** | `vague_unresolvable_followup` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q28** | `vague_unresolvable_followup` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q29** | `vague_unresolvable_followup` | **0.0000** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |
| **Q30** | `vague_unresolvable_followup` | **0.0001** | 0 | `answer` | ❌ No | ❌ No | I couldn't find this in your uploaded documents. W... |

## 4. Grounded Borderline / Stress-Test Evaluation (10 Questions)

Tested against `backend/evaluation/retrieval_questions_v2_borderline.json`:
- **B01-B05**: Challenging, formulaic, and highly-paraphrased answerable questions probing threshold survival.
- **B06-B10**: Hard negative unanswerable queries sharing dense domain terminology with the notes.

| ID | Type | Topic | Top-1 Rerank | Chunks Kept | Threshold Met? | Expected Behavior Achieved? |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **B01** | Answerable | Type 1 vs Type 2 Hypervisor Hardware Execution | **0.9447** | 3 | ✅ YES | ✅ Passed (Chunks retrieved) |
| **B02** | Answerable | Canny Hysteresis Dual Thresholding | **0.9736** | 4 | ✅ YES | ✅ Passed (Chunks retrieved) |
| **B03** | Answerable | Newton Method Hessian Inversion O(n³) Complexity | **0.9905** | 4 | ✅ YES | ✅ Passed (Chunks retrieved) |
| **B04** | Answerable | Affine vs Homography Parallel Line Preservation | **0.9921** | 2 | ✅ YES | ✅ Passed (Chunks retrieved) |
| **B05** | Answerable | Strong Duality and Slater's Condition | **0.9985** | 4 | ✅ YES | ✅ Passed (Chunks retrieved) |
| **B06** | Hard Negative | Borderline Negative: Cloud VXLAN 24-bit VNI Encapsulation | **0.0015** | 0 | ❌ No | ✅ Passed (0 chunks, Not Covered) |
| **B07** | Hard Negative | Borderline Negative: Vision Transformer Linear Patch Projection | **0.0003** | 0 | ❌ No | ✅ Passed (0 chunks, Not Covered) |
| **B08** | Hard Negative | Borderline Negative: Adam Optimizer Beta Moments | **0.0049** | 0 | ❌ No | ✅ Passed (0 chunks, Not Covered) |
| **B09** | Hard Negative | Borderline Negative: RetinaNet Focal Loss Hyperparameters | **0.0001** | 0 | ❌ No | ✅ Passed (0 chunks, Not Covered) |
| **B10** | Hard Negative | Borderline Negative: ADMM Primal Variable Splitting | **0.0153** | 0 | ❌ No | ✅ Passed (0 chunks, Not Covered) |

## 5. Web Search Behavior & Frontend Follow-up

1. **Decoupled Fallback**: Off-topic queries in knowledge base chats no longer route automatically to web search.
2. **Opt-in Request Flag**: Added `allow_web_search: bool = False` to `ChatRequest` (backend schema) and `AgentState` (LangGraph).
3. **Conversational Follow-Up**: The honest response (`I couldn't find this in your uploaded documents. Would you like me to search the web instead?`) combined with the updated supervisor prompt allows users to reply `Yes, search the web` to trigger web search.
4. **Frontend Action Required**: When building the UI toggle/button for Issue #9, the frontend chat client (`frontend/src/api/chatApi.js`) can pass `{ allow_web_search: true }` in the POST body to enable web search fallback directly.

## 6. Environment & Settings Metadata

- **Git Commit Hash**: `7cd8105950f58d2f59ec4a2a41e717c3c4783f64`
- **Timestamp**: `2026-11-05T19:04:14.216136+00:00`
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (threshold: `0.3`)
- **Embedding Model**: `gemini-embedding-2` (threshold: `0.68`)
- **Database Transaction**: Strictly read-only (`SET TRANSACTION READ ONLY`)