# Agentix Retrieval Evaluation Framework

This directory houses the ground-truth evaluation datasets, retrieval evaluation scripts, and experimental benchmarks for the Agentix RAG retrieval pipeline.

---

## Evaluation Datasets

### Version 1: `retrieval_questions.json` (Single-Document Baseline)
* **Scope**: Knowledge Base `Cloud Computing Notes` (`54335692-b9bd-4693-b074-5b536bdd51af`), single document `Cloud_Computing_Notes_Module1_2.docx` (`607a4e68-8135-4686-a60b-5970d65fbada`).
* **Total Chunks**: 97 chunks.
* **Total Questions**: 20 (12 Answerable, 8 Unanswerable).
* **Unanswerable Breakdown**: 2 Off-Topic (CV), 4 Same-Subject Missing Details, 2 Vague Follow-ups.
* **Benchmark Results**: Baseline Hit Rate@4 = 100.0% across all strategies.

### Version 2: `retrieval_questions_v2.json` (Multi-Document Corpus Expansion)
* **Scope**: Knowledge Base `Cloud Computing Notes` (`54335692-b9bd-4693-b074-5b536bdd51af`), containing 3 distinct syllabus documents:
  1. `Cloud_Computing_Notes_Module1_2.docx` (`607a4e68-8135-4686-a60b-5970d65fbada`, 97 chunks)
  2. `Computer_Vision_Notes_Module1_2.docx` (`f9ae765d-6b31-4bfe-8855-75d1627cd66f`, 72 chunks)
  3. `Optimization_for_ML_Notes_Module1_2.docx` (`927c37b8-4608-450d-9ff9-84cd52edeaa1`, 77 chunks)
* **Total Chunks**: 246 chunks (`chunk_index` 0 to 245 across 3 documents).
* **Total Questions**: 30 questions:
  * **18 Answerable Questions**: Exactly 6 grounded in each of the three documents (Q01–Q18).
  * **12 Unanswerable Questions**: 4 Off-Topic, 4 Same-Subject Missing Details, 4 Vague Follow-ups (Q19–Q30).
  * **Grounding Guarantee**: Every answerable question is mapped to verified document IDs, chunk IDs, page numbers, and source excerpts. Every unanswerable question has been verified to lack supporting evidence across all three documents.

---

## Evaluation Metrics

1. **Hit Rate@1**: Proportion of queries where a complete ground-truth evidence set is retrieved at rank #1.
2. **Hit Rate@4**: Proportion of queries where a complete ground-truth evidence set appears within the top 4 retrieved results.
3. **MRR@4 (Mean Reciprocal Rank @ 4)**: Reciprocal rank ($1 / \text{rank}$) of the completed ground-truth evidence set within top 4 ($0.0$ if unretrieved in top 4).
4. **Binary nDCG@4 (Normalized Discounted Cumulative Gain @ 4)**: Binary relevance position discounting ($1.0 / \log_2(\text{rank} + 1)$ for rank $\le 4$, with $\text{IDCG}=1.0$) evaluating rank quality within top 4.
5. **Alternative & Multi-Chunk Evidence Sets**: Queries support `expected_evidence_sets`:
   * **Multi-chunk sets**: When a question requires multiple distinct passages to be answered completely, completion rank is $\max_{c \in S} \text{rank}(c)$. Partial retrieval of the set does not count as a hit.
   * **Alternative sets**: When multiple different chunks or chunk combinations can independently answer the question, the overall completion rank is $\min_{S \text{ completed}} \text{rank}(S)$.
6. **Per-Document Performance Breakdown**: Separate evaluation metrics for each document in the multi-document corpus.
7. **Empirical Reranker Confidence & Unanswerable Score Tracking**: Analysis of maximum dense similarity and cross-encoder reranker confidence scores on unanswerable queries. Note that sigmoid-transformed cross-encoder logits are empirical monotonic scores for ranking, not calibrated posterior probabilities of answerability.

---

## Running the Evaluation

To evaluate against the expanded multi-document corpus:

```bash
cd backend
python scripts/eval_retrieval.py --dataset evaluation/retrieval_questions_v2.json --output-dir evaluation/results/expanded_corpus_v2
```

To evaluate the Version 1 dataset against the expanded corpus for comparison:

```bash
cd backend
python scripts/eval_retrieval.py --dataset evaluation/retrieval_questions.json --output-dir evaluation/results/v1_on_expanded_corpus
```

Results are exported to:
* `eval_summary.json`: High-level metric summary, per-document breakdowns, and reranker comparisons.
* `eval_detailed_results.json`: Complete ranking outputs for every query and strategy (`dense_rank`, `sparse_rank`, `rrf_score`, `rerank_score`, `strategy_rank`).
* `retrieval_eval_report.md`: Comprehensive Markdown report comparing baseline single-doc, V1 expanded, and V2 expanded runs.
