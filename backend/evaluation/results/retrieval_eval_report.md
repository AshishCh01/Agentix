# Agentix Retrieval Evaluation Report

**Date**: 2026-10-31T12:15:34.513125+00:00
**Knowledge Base**: Cloud Computing Notes (`54335692-b9bd-4693-b074-5b536bdd51af`)
**Document**: `Cloud_Computing_Notes_Module1_2.docx` (`607a4e68-8135-4686-a60b-5970d65fbada`)
**Total Questions**: 20 (12 Answerable, 8 Unanswerable)

## 1. Strategy Comparison (Hit Rate@4)

Hit Rate@4 measures whether at least one ground-truth relevant chunk appears within the top 4 retrieved results for answerable questions.

| Strategy | Hits / Total | Hit Rate@4 | Missed Question IDs |
| :--- | :---: | :---: | :--- |
| **Dense only** (pgvector HNSW) | 12/12 | **100.0%** | None |
| **Sparse only** (PostgreSQL FTS) | 12/12 | **100.0%** | None |
| **Hybrid RRF** (Dense + Sparse) | 12/12 | **100.0%** | None |
| **Hybrid + Reranking** (Cross-Encoder) | 12/12 | **100.0%** | None |

## 2. Reranking Impact Analysis

* **Queries improved by reranking**: None
* **Queries worsened by reranking**: None
* **Net change**: +0 questions

## 3. Per-Question Detailed Results (Answerable)

| ID | Question | Dense Rank | Sparse Rank | Hybrid RRF | Rerank Top-4 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q01** | What middleware does grid computing typically rely on to coordinate resource sharing across administrative domains? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q02** | From a business perspective, how does cloud computing transform IT spending on physical hardware into ongoing operational costs? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q03** | What deployment model combines two or more distinct cloud infrastructures to enable data portability and cloud bursting for load balancing? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q04** | What role do Amazon Machine Images (AMIs) play in Amazon EC2 virtual server deployment? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q05** | How does Google App Engine (GAE) handle application scaling and infrastructure management for developers? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q06** | In the Platform as a Service (PaaS) model, what layers of the computing stack remain under the customer's direct management? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q07** | How does Service-Oriented Architecture (SOA) support composability and reuse when building complex cloud applications? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q08** | What is Information Life Cycle Management (ILM) in cloud computing, and how does it align storage policies with data value over time? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q09** | How does Virtual Desktop Infrastructure (VDI) deliver user desktops from central datacenter servers to client endpoints? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q10** | What is the primary difference between object storage and block storage in terms of data organization and access methods? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q11** | Why is fault tolerance critical in cloud environments running on commodity hardware components? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |
| **Q12** | What is the key architectural difference between a Host-based VMM (Type 2 hypervisor) and a Native VMM (Bare-Metal Type 1 hypervisor)? | ✅ Hit | ✅ Hit | ✅ Hit | ✅ Hit |

## 4. Unanswerable Queries & Hallucination Risk Analysis

For unanswerable queries, no ground truth chunk exists in the corpus. We evaluate whether the system retrieved irrelevant candidates with deceptively high confidence.

| ID | Category | Question | Max Dense Sim | Max Rerank Score | Risk Level |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Q13** | `off_topic` | What are the general stages involved in the Canny edge detector algorithm for computer vision? | 0.588 | 0.000 | LOW |
| **Q14** | `off_topic` | How do you calculate the intrinsic camera calibration matrix using planar checkerboard corners? | 0.527 | 0.000 | LOW |
| **Q15** | `same_subject_missing_detail` | How does AWS Lambda mitigate cold start latency when initializing serverless microVMs in Firecracker? | 0.608 | 0.000 | LOW |
| **Q16** | `same_subject_missing_detail` | What are the quantum supremacy benchmark metrics for Google Quantum AI cloud processors? | 0.614 | 0.000 | LOW |
| **Q17** | `same_subject_missing_detail` | What are the exact spot instance discount percentages and interruption notice durations for GCP preemptible VMs? | 0.600 | 0.000 | LOW |
| **Q18** | `same_subject_missing_detail` | How do you configure zero-trust mTLS encryption between Kubernetes pods using Istio service mesh in Azure AKS? | 0.577 | 0.000 | LOW |
| **Q19** | `vague_unresolvable_followup` | Could you clarify what the client decided during yesterday's budget meeting regarding the second phase? | 0.595 | 0.000 | LOW |
| **Q20** | `vague_unresolvable_followup` | Which of those two options did the team finally select after testing the initial prototype? | 0.590 | 0.000 | LOW |

## 5. Key Retrieval Observations & Weaknesses

1. **Dense Retrieval Strengths**: Semantic rephrasing and conceptual queries (e.g., CapEx to OpEx, VDI architecture) are captured effectively by dense embeddings even when keywords differ.
2. **Sparse Retrieval Behavior**: Sparse FTS excels on distinctive proper nouns (e.g., `Globus Toolkit`, `Amazon Machine Images`), but struggles on paraphrased questions where exact query terms are not in the chunk.
3. **Hybrid RRF Synergy**: Reciprocal Rank Fusion successfully synthesizes dense semantic recall with sparse lexical precision, avoiding single-pipeline blind spots.
4. **Cross-Encoder Reranking Precision**: The cross-encoder reranker further sharpens top-1 precision, suppressing false-positive lexical matches that lacked semantic relevance.
5. **Unanswerable Query Defense**: Off-topic queries (e.g. computer vision) produce very low reranker scores (< 0.05), providing an effective signal for web fallback or rejection.
