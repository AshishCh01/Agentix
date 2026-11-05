# Agentix Baseline Relevance & Routing Evaluation (Issue #9 - Phase 1)

> **Measurement Only**: This report establishes the empirical baseline for Issue #9 ('No relevance threshold: off-topic questions get irrelevant chunks and web search fires wrongly'). No application behavior or retrieval code was modified.

## 1. Execution Settings & Environment

| Parameter | Value |
| :--- | :--- |
| **Date / Timestamp** | `2026-11-05T18:14:42.531578+00:00` |
| **Git Commit Hash** | `f1157e1a1973da9478d4871c4412298db5c59f5d` |
| **Evaluation Dataset** | `C:/Users/Ashish Choudhary/Desktop/Agentic RAG/backend/evaluation/retrieval_questions_v2.json` (v2.0) |
| **Knowledge Base Scope** | `Cloud Computing Notes` (`54335692-b9bd-4693-b074-5b536bdd51af`) |
| **Documents in Scope** | 3 documents (246 total chunks) |
| **Cutoff Metric (top_k)** | `4` |
| **Embedding Model** | `gemini-embedding-2` (2000 dims) |
| **Rerank Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (pool: 20) |
| **RRF Constant (k)** | `60` (retrieval_list_size: 30) |
| **HNSW Settings** | `ef_search=100`, `iterative_scan=relaxed_order` |
| **Database Transaction** | Strictly read-only (`SET TRANSACTION READ ONLY`) |

## 2. Retrieval Strategy Metrics Verification

Verification against PR #49 / PR #50 benchmark results across all 18 answerable queries:

| Strategy | Hit Rate@1 | Hit Rate@4 | MRR@4 | Binary nDCG@4 | PR #50 Match? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense only** (pgvector HNSW) | 55.56% (10/18) | **100.0%** (18/18) | **0.7593** | **0.8214** | ✅ Matches PR #50 |
| **Sparse only** (PostgreSQL FTS) | 61.11% (11/18) | **83.3%** (15/18) | **0.6944** | **0.7291** | ✅ Matches PR #50 |
| **Hybrid RRF** (Dense + Sparse) | 83.33% (15/18) | **94.4%** (17/18) | **0.8796** | **0.8962** | ✅ Matches PR #50 |
| **Hybrid + Reranking** (Cross-Encoder) | 66.67% (12/18) | **100.0%** (18/18) | **0.8056** | **0.8552** | ✅ Matches PR #50 |

## 3. Key Table for Issue #9: Per-Question Relevance & Routing Status

For each of the 30 queries in the V2 dataset, this table presents:
- **Top-1 Rerank Score**: Output of `cross-encoder/ms-marco-MiniLM-L-6-v2` squashed by sigmoid [0, 1].
- **Top-1 Dense Sim**: Cosine similarity (1 - cosine distance) from `gemini-embedding-2`.
- **Would Web Search Fire?**: Whether the current app logic triggers web search.
- **Would Irrelevant Chunks Pass?**: Whether chunks unrelated to the query pass to the answer prompt.

| ID | Category | Question | Top-1 Rerank Score | Top-1 Dense Sim | Web Search Fires? | Irrelevant Chunks Passed? | Top Retrieved Document |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Q01** | `answerable` | What middleware does grid computing typically rely on to coordinate resource sharing across administrative domains? | **0.9997** | 0.8090 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q02** | `answerable` | From a business perspective, how does cloud computing transform IT spending on physical hardware into ongoing operational costs? | **0.9964** | 0.7640 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q03** | `answerable` | What is the distinction between vertical elasticity and horizontal elasticity in cloud computing systems? | **0.9989** | 0.8215 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q04** | `answerable` | How does Information Life Cycle Management (ILM) adjust storage tiers as data ages from initial creation? | **0.9900** | 0.8003 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q05** | `answerable` | Why is fault tolerance essential in cloud computing environments where commodity hardware is deployed? | **0.9998** | 0.7974 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q06** | `answerable` | What is server virtualization and what role does the hypervisor play in managing physical server resources? | **0.9986** | 0.7660 | ❌ No | ❌ No | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q07** | `answerable` | What operations characterize low-level computer vision compared to higher levels of abstraction? | **0.9984** | 0.8475 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q08** | `answerable` | In Canny edge detection, how does non-maximum suppression produce thin single-pixel wide boundaries? | **0.9938** | 0.7591 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q09** | `answerable` | Under the pinhole camera model, why do 3D objects that are farther away appear smaller on the image plane? | **0.9997** | 0.7673 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q10** | `answerable` | How does Zhang's method utilize multiple planar checkerboard views to compute intrinsic camera matrix K and lens distortion? | **0.9987** | 0.8274 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q11** | `answerable` | How do autonomous vehicle perception systems combine camera, LiDAR, and RADAR data to maintain a unified 3D scene model? | **0.9878** | 0.7684 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q12** | `answerable` | What are the core stages in a deep face recognition pipeline between raw frame detection and database matching? | **0.8865** | 0.7961 | ❌ No | ❌ No | `Computer_Vision_Notes_Module1_2.docx` |
| **Q13** | `answerable` | What theoretical approximation guarantee do greedy algorithms provide when optimizing monotone submodular functions for data subset selection? | **0.9904** | 0.8232 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q14** | `answerable` | What is the mathematical definition of a convex function, and why are local minima desirable in convex optimization? | **0.9302** | 0.8125 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q15** | `answerable` | Under what conditions does standard gradient descent guarantee convergence to the global minimum of a differentiable function? | **0.9977** | 0.7860 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q16** | `answerable` | How does Nesterov's Accelerated Gradient (NAG) achieve an optimal convergence rate of O(1/k²) compared to standard gradient descent? | **0.9999** | 0.8332 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q17** | `answerable` | Why is L-BFGS preferred over exact Newton's method for medium-scale machine learning problems? | **0.9968** | 0.8252 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q18** | `answerable` | What is the relationship between weak duality and strong duality under Slater's condition in convex optimization? | **0.9992** | 0.8239 | ❌ No | ❌ No | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q19** | `off_topic` | What anatomical features differentiate Theropod dinosaur pelvic structures from Ornithischian dinosaurs in vertebrate paleontology? | **0.0000** | 0.5470 | ❌ No | ⚠️ **YES** | `Computer_Vision_Notes_Module1_2.docx` |
| **Q20** | `off_topic` | How does the Hatch-Slack C4 photosynthetic pathway spatially separate carbon fixation from the Calvin cycle in plant leaves? | **0.0000** | 0.5438 | ❌ No | ⚠️ **YES** | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q21** | `off_topic` | What seismic wave velocity anomalies characterize the boundary layer between the Earth's mantle and outer core at the D-double-prime discontinuity? | **0.0000** | 0.5637 | ❌ No | ⚠️ **YES** | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q22** | `off_topic` | How did 18th-century Baroque composers use invertible counterpoint and equal temperament tuning in polyphonic fugues? | **0.0000** | 0.5364 | ❌ No | ⚠️ **YES** | `Computer_Vision_Notes_Module1_2.docx` |
| **Q23** | `same_subject_missing_detail` | How do you configure eBPF kernel bypass programs in Cilium to replace kube-proxy iptables routing in Kubernetes clusters? | **0.0000** | 0.5562 | ❌ No | ⚠️ **YES** | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q24** | `same_subject_missing_detail` | What are the exact architectural modifications and loss functions used in the YOLOv10 dual-label assignment head for end-to-end NMS-free object detection? | **0.0001** | 0.5724 | ❌ No | ⚠️ **YES** | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q25** | `same_subject_missing_detail` | What are the specific PyTorch Fully Sharded Data Parallel (FSDP) ZeRO-3 memory buffer management formulas for 70B parameter models? | **0.0000** | 0.5741 | ❌ No | ⚠️ **YES** | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q26** | `same_subject_missing_detail` | How do Neural Radiance Fields (NeRF) compute volume density gradients along camera rays using hierarchical stratified sampling? | **0.0060** | 0.6040 | ❌ No | ⚠️ **YES** | `Computer_Vision_Notes_Module1_2.docx` |
| **Q27** | `vague_unresolvable_followup` | Could you send over the updated figures and benchmark numbers from yesterday's second test run? | **0.0000** | 0.5872 | ❌ No | ⚠️ **YES** | `Optimization_for_ML_Notes_Module1_2.docx` |
| **Q28** | `vague_unresolvable_followup` | Did the committee approve the revised proposal that was submitted before the end of the quarter? | **0.0000** | 0.5903 | ❌ No | ⚠️ **YES** | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q29** | `vague_unresolvable_followup` | Which of those three recommendations should our engineering team prioritize for the sprint? | **0.0000** | 0.5974 | ❌ No | ⚠️ **YES** | `Cloud_Computing_Notes_Module1_2.docx` |
| **Q30** | `vague_unresolvable_followup` | What was the final outcome after applying the workaround discussed in the previous section? | **0.0001** | 0.6171 | ❌ No | ⚠️ **YES** | `Cloud_Computing_Notes_Module1_2.docx` |

## 4. Gap Analysis

The gap analysis measures whether a clean separation margin exists between the lowest-scoring answerable query and the highest-scoring unanswerable query.

### 4.1 Cross-Encoder Rerank Score Gap Analysis

- **Lowest Answerable Top-1 Rerank Score**: **0.8865** (Query: `Q12` in `Computer_Vision_Notes_Module1_2.docx`)
  - *Question*: "What are the core stages in a deep face recognition pipeline between raw frame detection and database matching?"
- **Highest Unanswerable Top-1 Rerank Score**: **0.0060** (Query: `Q26`, Category: `same_subject_missing_detail`)
  - *Question*: "How do Neural Radiance Fields (NeRF) compute volume density gradients along camera rays using hierarchical stratified sampling?"
- **Clean Gap Exists?**: **✅ YES**
- **Separation Margin (Gap Size)**: **+0.8806**
- **Overlapping Questions**: **None** (Clean separation across all 30 queries).

#### Per-Document Answerable Minimums (Rerank Score)

| Document | Filename | Lowest Query | Min Score | Mean Score |
| :--- | :--- | :---: | :---: | :---: |
| `607a4e68...` | `Cloud_Computing_Notes_Module1_2.docx` | `Q04` | **0.9900** | 0.9972 |
| `927c37b8...` | `Optimization_for_ML_Notes_Module1_2.docx` | `Q14` | **0.9302** | 0.9857 |
| `f9ae765d...` | `Computer_Vision_Notes_Module1_2.docx` | `Q12` | **0.8865** | 0.9775 |

#### Unanswerable Breakdown by Category (Rerank Score)

| Category | Query Count | Max Top-1 Score | Mean Top-1 Score | Max Query ID |
| :--- | :---: | :---: | :---: | :---: |
| `off_topic` | 4 | **0.000020** | 0.000018 | `Q20` |
| `same_subject_missing_detail` | 4 | **0.005960** | 0.001536 | `Q26` |
| `vague_unresolvable_followup` | 4 | **0.000054** | 0.000026 | `Q30` |

### 4.2 Dense Similarity Gap Analysis

- **Lowest Answerable Top-1 Dense Similarity**: **0.7591** (Query: `Q08` in `Computer_Vision_Notes_Module1_2.docx`)
  - *Question*: "In Canny edge detection, how does non-maximum suppression produce thin single-pixel wide boundaries?"
- **Highest Unanswerable Top-1 Dense Similarity**: **0.6171** (Query: `Q30`, Category: `vague_unresolvable_followup`)
  - *Question*: "What was the final outcome after applying the workaround discussed in the previous section?"
- **Clean Gap Exists?**: **✅ YES**
- **Separation Margin (Gap Size)**: **+0.1420**

#### Per-Document Answerable Minimums (Dense Similarity)

| Document | Filename | Lowest Query | Min Dense Sim | Mean Dense Sim |
| :--- | :--- | :---: | :---: | :---: |
| `607a4e68...` | `Cloud_Computing_Notes_Module1_2.docx` | `Q02` | **0.7640** | 0.7930 |
| `927c37b8...` | `Optimization_for_ML_Notes_Module1_2.docx` | `Q15` | **0.7860** | 0.8173 |
| `f9ae765d...` | `Computer_Vision_Notes_Module1_2.docx` | `Q08` | **0.7591** | 0.7943 |

## 5. Web Search & Chunk-Passing Analysis

### 5.1 Root Cause Architecture Inspection
1. **Dense Search Nearest-Neighbor Nature**: In `ai/services/retrieval_service.py`, dense pgvector search retrieves the `LIMIT :list_size` (30) nearest chunks via `<=>` cosine distance regardless of absolute relevance. It never returns zero chunks if any indexed chunks exist in the scoped knowledge base.
2. **Missing Cutoff in Vector Tool**: `vector_search_tool` (`ai/tools/vector_search.py`) returns all top 4 reranked chunks directly into `state['retrieved_chunks']` and formats `state['formatted_context']` with zero score filtering.
3. **Inappropriate Web Search Fallback Condition**: `route_after_vector_search` (`ai/agents/graph.py`) only routes to `web_search` when `not retrieved_chunks or len(retrieved_chunks) == 0`. Since vector search unconditionally returns 4 chunks on a populated corpus, this fallback never fires on off-topic questions. However, if a document is currently `processing` (0 chunks indexed), this fallback wrongly fires and exposes private queries to the public web.
4. **Hallucination Exposure in Answer Agent**: `run_answer_agent` (`ai/agents/answer.py`) receives 4 completely irrelevant chunks for every unanswerable query (Q19-Q30) and attempts to synthesize an answer or hallucinates from training data.

### 5.2 Summary of Measured Behavior Across All 30 Questions
- **Answerable Queries (Q01-Q18)**: Web search fires: 0/18 (0.0%). Irrelevant chunks exclusively passed: 0/18 (0.0%). All 18 retrieve ground-truth chunks in top-4.
- **Unanswerable Queries (Q19-Q30)**: Web search fires: 0/12 (0.0%). Irrelevant chunks passed: **12/12 (100.0%)**.
  - For all 12 unanswerable queries, 4 completely irrelevant chunks are injected into the LLM system prompt.

## 6. Recommendations for Phase 2

> [!IMPORTANT]
> **Recommendation for Phase 2**: The empirical findings strongly support introducing relevance thresholding based primarily on **Cross-Encoder Rerank Score**, with **Dense Similarity** as a secondary fallback when reranking is disabled (`RERANK_PROVIDER == 'none'`).

1. **Primary Signal (Rerank Score)**: Cross-encoder rerank score exhibits a **clean separation gap of 0.8806** (min answerable: `0.8865`, max unanswerable: `0.0060`).
   - **Recommended `MIN_RELEVANCE_RERANK` Value**: **`0.30`** (or a safe range of `[0.10, 0.50]`). A threshold of 0.30 sits comfortably between the highest unanswerable score (< 0.01) and lowest answerable score (0.95+), leaving an enormous safety margin.
2. **Secondary Signal (Dense Similarity Fallback)**: Dense similarity achieves a positive clean separation margin of +0.1420 (min answerable: `0.7591`, max unanswerable: `0.6171`), but with a significantly narrower safety buffer than Cross-Encoder reranking (+0.1420 vs +0.8806).
   - If reranking is disabled, a candidate dense similarity threshold around `0.68` (e.g. range `[0.65, 0.70]`, positioned between unanswerable max `0.6171` and answerable min `0.7591`) provides viable separation, but with much lower tolerance to paraphrasing variation than the Cross-Encoder.
3. **Zero-Chunk / Processing State Handling**: When `route_after_vector_search` finds 0 chunks passing `MIN_RELEVANCE`: check if documents in the KB have `status == 'processing'`. If so, respond with 'Document indexing in progress' rather than searching the web.
4. **Web Search Decoupling**: Make web search an explicit router decision (from the Supervisor) rather than a fallback for missing document content.