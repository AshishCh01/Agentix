# Agentix RAG Backend Evaluation Report

> **Generated at**: 2026-11-19 18:09:52 UTC
> **Pipeline Configuration**: 30 vector candidates, RRF $k=60$, 20 rerank candidates, 4 final chunks, Cross-Encoder threshold >= 0.30.

## Dataset: `retrieval_questions_v2`

- **Total Questions**: 30 (18 answerable, 12 unanswerable)
- **Failed Questions**: 0

### 1. Retrieval Performance: Before vs. After Reranker (Cutoff K=4, Answerable Only)

| Metric | Stage (a): Hybrid RRF Top-4 (Before Reranker) | Stage (b): Cross-Encoder Final-4 (After Reranker & Threshold) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Hit Rate@1** | 83.3% | **66.7%** | -16.7% |
| **Hit Rate@4** | 94.4% | **100.0%** | +5.6% |
| **MRR@4** | 0.8796 | **0.8056** | -0.0741 |
| **Binary nDCG@4** | 0.8962 | **0.8552** | -0.0410 |
| **Precision@4** | 0.2361 | **0.2500** | +0.0139 |
| **Recall@4** | 0.9444 | **1.0000** | +0.0556 |
| **MAP@4** | 0.8796 | **0.8056** | -0.0741 |

### 2. Abstention, Hallucination & Answer Quality

| Metric Category | Metric Name | Value | Evaluation Basis / Target |
| :--- | :--- | :---: | :--- |
| **Abstention** | Correct Abstention Rate | **100.0%** | Unanswerable queries where system declined (target: 100%) |
| **Abstention** | False Abstention Rate | **0.0%** | Answerable queries wrongly declined (target: 0%) |
| **Abstention** | Overall Abstention Accuracy | **100.0%** | Correct routing across all 30 queries |
| **Hallucination** | Hallucination Rate | **0.0%** | Unanswerable queries answered confidently (target: 0%) |
| **Answer Quality** | Faithfulness (LLM Judge) | **100.0%** | Answer claims strictly supported by retrieved chunks |
| **Answer Quality** | Answer Correctness (LLM Judge) | **100.0%** | Answer factually matches ground-truth source passage |

### 3. Context Quality & Relevance Separation

| Metric Name | Value | Description |
| :--- | :---: | :--- |
| **Context Precision** | **0.2500** (25.0%) | Average fraction of top 4 chunks that are relevant |
| **Avg Top-1 Rerank (Answerable)** | **0.9868** | Cross-encoder confidence on answerable queries |
| **Avg Top-1 Rerank (Unanswerable)** | **0.0005** | Cross-encoder score on unanswerable queries |
| **Threshold Separation Margin** | **+0.9863** | Separation gap around the 0.30 threshold |
| **Avg Final Chunks Passing >= 0.30 (Answerable)** | **3.33 / 4** | Final surviving document chunks passed to answer agent (max 4) |
| **Avg Pool Chunks Passing >= 0.30 (Answerable)** | **4.89 / 20** | Surviving chunks from candidate pool before top-4 truncation |
| **Avg Final Chunks Passing >= 0.30 (Unanswerable)** | **0.00 / 4** | Final leakage rate of irrelevant chunks (target: 0.0) |
| **Avg Pool Chunks Passing >= 0.30 (Unanswerable)** | **0.00 / 20** | Leakage in candidate pool before top-4 truncation (target: 0.0) |

### 4. System Performance & Agent Reliability

| Operational Metric | Value | Details |
| :--- | :---: | :--- |
| **Retrieval Latency p50** | **4703.1 ms** | Hybrid search + cross-encoder rerank median latency |
| **Retrieval Latency p95** | **7403.9 ms** | Hybrid search + cross-encoder rerank 95th percentile |
| **End-to-End Latency p50** | **12.31 s** | Full agent graph execution median latency |
| **End-to-End Latency p95** | **24.49 s** | Full agent graph execution 95th percentile |
| **Average Retries Used** | **0.00** | Self-correction reflection retry loop count |
| **Answers Marked Unverified** | **0.0%** | Responses flagged with unverified caveat |
| **Average Answer Length (Words, Answerable)** | **87.5 words** | Average words per answer for answerable queries |
| **Average Answer Length (Words, All)** | **59.3 words** | Average words per response across all queries |
| **Average Tokens Per Answer** | **N/A (LangGraph AgentState and llm_service do not aggregate token usage metadata)** | |

### 5. Detailed Failure Analysis

🎉 **Zero failures! All questions satisfied retrieval, abstention, and answer quality criteria.**

---

## Dataset: `retrieval_questions_v2_borderline`

- **Total Questions**: 10 (5 answerable, 5 unanswerable)
- **Failed Questions**: 0

### 1. Retrieval Performance: Before vs. After Reranker (Cutoff K=4, Answerable Only)

| Metric | Stage (a): Hybrid RRF Top-4 (Before Reranker) | Stage (b): Cross-Encoder Final-4 (After Reranker & Threshold) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Hit Rate@1** | 80.0% | **60.0%** | -20.0% |
| **Hit Rate@4** | 100.0% | **100.0%** | +0.0% |
| **MRR@4** | 0.8500 | **0.8000** | -0.0500 |
| **Binary nDCG@4** | 0.8861 | **0.8524** | -0.0338 |
| **Precision@4** | 0.2500 | **0.2500** | +0.0000 |
| **Recall@4** | 1.0000 | **1.0000** | +0.0000 |
| **MAP@4** | 0.8500 | **0.8000** | -0.0500 |

### 2. Abstention, Hallucination & Answer Quality

| Metric Category | Metric Name | Value | Evaluation Basis / Target |
| :--- | :--- | :---: | :--- |
| **Abstention** | Correct Abstention Rate | **100.0%** | Unanswerable queries where system declined (target: 100%) |
| **Abstention** | False Abstention Rate | **0.0%** | Answerable queries wrongly declined (target: 0%) |
| **Abstention** | Overall Abstention Accuracy | **100.0%** | Correct routing across all 10 queries |
| **Hallucination** | Hallucination Rate | **0.0%** | Unanswerable queries answered confidently (target: 0%) |
| **Answer Quality** | Faithfulness (LLM Judge) | **100.0%** | Answer claims strictly supported by retrieved chunks |
| **Answer Quality** | Answer Correctness (LLM Judge) | **100.0%** | Answer factually matches ground-truth source passage |

### 3. Context Quality & Relevance Separation

| Metric Name | Value | Description |
| :--- | :---: | :--- |
| **Context Precision** | **0.2500** (25.0%) | Average fraction of top 4 chunks that are relevant |
| **Avg Top-1 Rerank (Answerable)** | **0.9799** | Cross-encoder confidence on answerable queries |
| **Avg Top-1 Rerank (Unanswerable)** | **0.0044** | Cross-encoder score on unanswerable queries |
| **Threshold Separation Margin** | **+0.9755** | Separation gap around the 0.30 threshold |
| **Avg Final Chunks Passing >= 0.30 (Answerable)** | **3.40 / 4** | Final surviving document chunks passed to answer agent (max 4) |
| **Avg Pool Chunks Passing >= 0.30 (Answerable)** | **3.80 / 20** | Surviving chunks from candidate pool before top-4 truncation |
| **Avg Final Chunks Passing >= 0.30 (Unanswerable)** | **0.00 / 4** | Final leakage rate of irrelevant chunks (target: 0.0) |
| **Avg Pool Chunks Passing >= 0.30 (Unanswerable)** | **0.00 / 20** | Leakage in candidate pool before top-4 truncation (target: 0.0) |

### 4. System Performance & Agent Reliability

| Operational Metric | Value | Details |
| :--- | :---: | :--- |
| **Retrieval Latency p50** | **4829.0 ms** | Hybrid search + cross-encoder rerank median latency |
| **Retrieval Latency p95** | **5175.5 ms** | Hybrid search + cross-encoder rerank 95th percentile |
| **End-to-End Latency p50** | **9.99 s** | Full agent graph execution median latency |
| **End-to-End Latency p95** | **14.48 s** | Full agent graph execution 95th percentile |
| **Average Retries Used** | **0.00** | Self-correction reflection retry loop count |
| **Answers Marked Unverified** | **0.0%** | Responses flagged with unverified caveat |
| **Average Answer Length (Words, Answerable)** | **56.4 words** | Average words per answer for answerable queries |
| **Average Answer Length (Words, All)** | **36.7 words** | Average words per response across all queries |
| **Average Tokens Per Answer** | **N/A (LangGraph AgentState and llm_service do not aggregate token usage metadata)** | |

### 5. Detailed Failure Analysis

🎉 **Zero failures! All questions satisfied retrieval, abstention, and answer quality criteria.**

---
