# Agentix Retrieval Evaluation Report

**Date**: 2026-10-31T17:05:00.442939+00:00
**Knowledge Base**: Cloud Computing Notes (`54335692-b9bd-4693-b074-5b536bdd51af`)
**Total Documents in Scope**: 1
**Document**: Cloud_Computing_Notes_Module1_2.docx (`607a4e68-8135-4686-a60b-5970d65fbada`)
**Total Indexed Chunks**: 97
**Questions Evaluated**: 20 (12 Answerable, 8 Unanswerable)

## 1. Overall Strategy Comparison

Metrics evaluated across answerable questions:
- **Hit Rate@1**: Proportion of queries where a complete evidence set is ranked #1.
- **Hit Rate@4**: Proportion of queries where at least one complete evidence set appears in the top 4.
- **MRR@4**: Mean Reciprocal Rank at cutoff 4 (1/rank for earliest complete evidence set; 0 if not in top 4).
- **Binary nDCG@4**: Normalized Discounted Cumulative Gain at cutoff 4 (logarithmic rank discounting with binary relevance).

| Strategy | Hit Rate@1 | Hit Rate@4 | MRR@4 | Binary nDCG@4 | Missed Questions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense only** (pgvector HNSW) | 50.0% (6/12) | **100.0%** (12/12) | **0.7361** | **0.8046** | None |
| **Sparse only** (PostgreSQL FTS) | 66.67% (8/12) | **100.0%** (12/12) | **0.7986** | **0.8494** | None |
| **Hybrid RRF** (Dense + Sparse) | 75.0% (9/12) | **100.0%** (12/12) | **0.8750** | **0.9077** | None |
| **Hybrid + Reranking** (Cross-Encoder) | 83.33% (10/12) | **100.0%** (12/12) | **0.9028** | **0.9276** | None |

## 2. Multi-Run Comparison (Baseline vs. Single-Doc vs. Multi-Doc)

Comparison of retrieval accuracy and rank metrics across the three experimental setups:

| Metric / Strategy | Run 1: V1 Initial Baseline (1 Doc, 97 chunks) | Run 2: V1 Single-Doc Evaluation (1 Doc, 97 chunks) | Run 3: V2 Multi-Doc (3 Docs, 246 chunks) |
| :--- | :---: | :---: | :---: |
| **Questions Evaluated** | 20 (12 Ans, 8 Unans) | 20 (12 Ans, 8 Unans) | 30 (18 Ans, 12 Unans) |
| **Dense Hit Rate@1** | N/A | 50.0% (6/12) | 55.6% (10/18) |
| **Dense Hit Rate@4** | 100.0% (12/12) | 100.0% (12/12) | 100.0% (18/18) |
| **Dense MRR@4** | N/A | 0.7361 | 0.7593 |
| **Dense Binary nDCG@4** | N/A | 0.8046 | 0.8214 |
| **Sparse Hit Rate@1** | N/A | 66.7% (8/12) | 61.1% (11/18) |
| **Sparse Hit Rate@4** | 100.0% (12/12) | 100.0% (12/12) | 83.3% (15/18) |
| **Sparse MRR@4** | N/A | 0.7986 | 0.6944 |
| **Sparse Binary nDCG@4** | N/A | 0.8494 | 0.7291 |
| **Hybrid RRF Hit Rate@1** | N/A | 75.0% (9/12) | 83.3% (15/18) |
| **Hybrid RRF Hit Rate@4** | 100.0% (12/12) | 100.0% (12/12) | 94.4% (17/18) |
| **Hybrid RRF MRR@4** | N/A | 0.8750 | 0.8796 |
| **Hybrid RRF Binary nDCG@4** | N/A | 0.9077 | 0.8962 |
| **Hybrid + Rerank Hit Rate@1** | N/A | 83.3% (10/12) | 66.7% (12/18) |
| **Hybrid + Rerank Hit Rate@4** | 100.0% (12/12) | 100.0% (12/12) | 100.0% (18/18) |
| **Hybrid + Rerank MRR@4** | N/A | 0.9028 | 0.8056 |
| **Hybrid + Rerank Binary nDCG@4** | N/A | 0.9276 | 0.8552 |

## 3. Per-Document Retrieval Performance

Evaluates retrieval effectiveness for the single document in scope:

| Document ID | Filename | Queries | Dense Hit@1 (Hit@4) | Sparse Hit@1 (Hit@4) | Hybrid RRF Hit@1 (Hit@4) | Rerank Hit@1 (Hit@4) | RRF MRR@4 | Rerank MRR@4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `607a4e68...` | `Cloud_Computing_Notes_Module1_2.docx` | 12 | 50.0% (100.0%) | 66.7% (100.0%) | **75.0% (100.0%)** | **83.3% (100.0%)** | 0.8750 | 0.9028 |

## 4. Cross-Encoder Reranking Impact Analysis

- **Queries improved by reranking**: None
- **Queries worsened by reranking**: None
- **Net change**: +0 questions

### Rank Shift & Precision Refinement
The cross-encoder reranker evaluates full query-document cross-attention pairs (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top-20 hybrid candidate pool. On this single-document corpus, cross-encoder reranking provides precision refinement on conceptual and architectural queries, boosting Hit@1 and MRR@4 by prioritizing the most relevant passages. All 12 answerable queries achieve 100% Hit@4 across all retrieval strategies.

## 5. Per-Question Detailed Results (Answerable)

| ID | Question | Expected Doc | Dense (Rank) | Sparse (Rank) | Hybrid RRF (Rank) | Rerank (Rank) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Q01** | What middleware does grid computing typically rely on to coordinate resource sharing across administrative domains? | `607a4e68` | ✅ Hit (#2) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q02** | From a business perspective, how does cloud computing transform IT spending on physical hardware into ongoing operational costs? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#4) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q03** | What deployment model combines two or more distinct cloud infrastructures to enable data portability and cloud bursting for load balancing? | `607a4e68` | ✅ Hit (#2) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q04** | What role do Amazon Machine Images (AMIs) play in Amazon EC2 virtual server deployment? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#2) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q05** | How does Google App Engine (GAE) handle application scaling and infrastructure management for developers? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q06** | In the Platform as a Service (PaaS) model, what layers of the computing stack remain under the customer's direct management? | `607a4e68` | ✅ Hit (#2) | ✅ Hit (#2) | ✅ Hit (#2) | ✅ Hit (#3) |
| **Q07** | How does Service-Oriented Architecture (SOA) support composability and reuse when building complex cloud applications? | `607a4e68` | ✅ Hit (#2) | ✅ Hit (#3) | ✅ Hit (#2) | ✅ Hit (#2) |
| **Q08** | What is Information Life Cycle Management (ILM) in cloud computing, and how does it align storage policies with data value over time? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q09** | How does Virtual Desktop Infrastructure (VDI) deliver user desktops from central datacenter servers to client endpoints? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q10** | What is the primary difference between object storage and block storage in terms of data organization and access methods? | `607a4e68` | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q11** | Why is fault tolerance critical in cloud environments running on commodity hardware components? | `607a4e68` | ✅ Hit (#2) | ✅ Hit (#1) | ✅ Hit (#1) | ✅ Hit (#1) |
| **Q12** | What is the key architectural difference between a Host-based VMM (Type 2 hypervisor) and a Native VMM (Bare-Metal Type 1 hypervisor)? | `607a4e68` | ✅ Hit (#3) | ✅ Hit (#1) | ✅ Hit (#2) | ✅ Hit (#1) |

## 6. Unanswerable Queries & Hallucination Risk Analysis

Unanswerable queries evaluate corpus isolation and defense against hallucinations across three distinct categories:
1. **Off-Topic**: Queries completely outside the single-document knowledge domain.
2. **Same-Subject Missing Detail**: Plausible queries within the subject area whose specific answers are absent from the notes.
3. **Vague / Unresolvable Follow-Up**: Conversational queries relying on external context without referents in the text.

> **Note on Score Interpretation**: The cross-encoder outputs raw logits squashed to $[0, 1]$ via the standard logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$. These values are **empirical monotonic relevance scores**, NOT calibrated posterior probabilities $P(\text{Relevant} \mid Q, D)$. They provide an effective heuristic ranking and thresholding signal, but must not be interpreted as probabilistic confidence levels without calibration (e.g., via Platt scaling or isotonic regression).

| ID | Category | Question | Max Dense Sim | Max Rerank Score | Retrieved Documents | Risk Level |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: |
| **Q13** | `off_topic` | What are the general stages involved in the Canny edge detector algorithm for computer vision? | 0.588 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q14** | `off_topic` | How do you calculate the intrinsic camera calibration matrix using planar checkerboard corners? | 0.527 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q15** | `same_subject_missing_detail` | How does AWS Lambda mitigate cold start latency when initializing serverless microVMs in Firecracker? | 0.608 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q16** | `same_subject_missing_detail` | What are the quantum supremacy benchmark metrics for Google Quantum AI cloud processors? | 0.614 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q17** | `same_subject_missing_detail` | What are the exact spot instance discount percentages and interruption notice durations for GCP preemptible VMs? | 0.600 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q18** | `same_subject_missing_detail` | How do you configure zero-trust mTLS encryption between Kubernetes pods using Istio service mesh in Azure AKS? | 0.577 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q19** | `vague_unresolvable_followup` | Could you clarify what the client decided during yesterday's budget meeting regarding the second phase? | 0.595 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |
| **Q20** | `vague_unresolvable_followup` | Which of those two options did the team finally select after testing the initial prototype? | 0.590 | 0.000 | `Cloud_Computing_Notes_Module1_2.docx` | LOW |

## 7. Single-Document Corpus Insights & Architecture Findings

1. **Single-Document Baseline Scope**: The evaluation targets a single technical document (Cloud_Computing_Notes_Module1_2.docx, 97 chunks) establishing core baseline retrieval performance.
2. **Corpus Isolation Defense**: The retrieval pipeline effectively scopes queries to the target document, properly isolating candidate generation from unrelated materials.
3. **Sparse Retrieval Sensitivity**: Sparse PostgreSQL FTS (ts_rank_cd) excels when exact terminology matches the text, while dense retrieval handles conceptual paraphrasing.
4. **Hybrid RRF Complementarity**: RRF balances lexical and semantic signals, consistently delivering 100% Hit@4 across all answerable questions.
5. **Cross-Encoder Score Discrimination**: Unanswerable queries yield substantially lower cross-encoder scores (mean: 0.000092, max: 0.000326; off-topic mean: 0.000015) than answerable queries (mean: 0.9906, min: 0.9492), establishing an empirical separation margin of 0.9489 as an effective gating signal for Issue #9's relevance thresholding.
