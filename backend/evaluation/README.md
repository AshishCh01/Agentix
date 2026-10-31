# Retrieval Evaluation Dataset

This dataset provides a rigorous, ground-truth evaluation benchmark for the Agentix RAG retrieval pipeline (Issue #33).

## Evaluation Scope

* **Knowledge Base ID**: `54335692-b9bd-4693-b074-5b536bdd51af`
* **Knowledge Base Name**: Cloud Computing Notes
* **Owner User ID**: `42e4af12-1cec-491d-9f83-08a11baaf9ec`
* **Document ID**: `607a4e68-8135-4686-a60b-5970d65fbada`
* **Document Filename**: `Cloud_Computing_Notes_Module1_2.docx`
* **Total Chunks in Corpus**: 97 chunks (`chunk_index` 0 to 96)
* **Embedding Model**: `gemini-embedding-2` (2000 dimensions, truncated for pgvector HNSW index)
* **Full-Text Tokenizer**: PostgreSQL `tsvector` (`to_tsvector('english', content)`) with GIN index

### Selection Rationale
The `Cloud Computing Notes` knowledge base was chosen because it represents a complete, high-quality development corpus containing 97 well-structured chunks spanning foundational distributed computing concepts, NIST definitions, cloud service models (IaaS/PaaS/SaaS), architecture paradigms (SOA, Cloud Stack, VDI, Storage), fault tolerance, and hypervisor architectures. It provides diverse lexical and semantic variety suitable for testing dense vector search, sparse keyword matching, reciprocal rank fusion, and cross-encoder reranking.

## Dataset Composition

The dataset comprises **20 questions** in total:

### 1. Answerable Questions (12 items: Q01 – Q12)
Every answerable question is strictly grounded in an actual indexed chunk from the development database:
* **Direct Lexical Queries**: Target specific terminology (e.g., Globus Toolkit in Grid Computing, Amazon Machine Images).
* **Paraphrased Conceptual Queries**: Rephrase underlying concepts without verbatim keyword copying (e.g., CapEx to OpEx business shift, VDI desktop streaming).
* **Semantic & Multi-Aspect Queries**: Require semantic understanding of definitions, architectural boundaries, and comparisons (e.g., PaaS customer management boundaries, Type 1 vs. Type 2 hypervisors, Object vs. Block storage).

Each answerable record includes:
* `id`: Unique identifier (e.g., `Q01`)
* `question`: Evaluation query string
* `answerable`: `true`
* `expected_document_id`: Target document UUID
* `expected_chunk_ids`: List of verified ground-truth chunk UUIDs
* `expected_page`: Document page reference (where available)
* `source_passage`: Verbatim excerpt from the indexed chunk confirming the answer

### 2. Unanswerable Questions (8 items: Q13 – Q20)
Used to evaluate retrieval behavior when user queries cannot or should not be satisfied by the indexed corpus:
* **Off-Topic Questions (Q13 – Q14)**: Topics entirely outside the domain (e.g., Computer Vision Canny edge detection, camera calibration matrix).
* **Same-Subject Missing Detail (Q15 – Q18)**: Questions within the cloud computing domain whose specific answers are absent from this syllabus document (e.g., AWS Lambda Firecracker cold starts, Google Quantum AI benchmarks, GCP preemptible VM discount tables, Kubernetes Istio mTLS manifests).
* **Vague / Unresolvable Follow-Ups (Q19 – Q20)**: Queries relying on ambiguous pronouns and ungrounded external meeting context.

Each unanswerable record includes:
* `id`: Unique identifier (e.g., `Q13`)
* `question`: Evaluation query string
* `answerable`: `false`
* `unanswerable_category`: `off_topic` | `same_subject_missing_detail` | `vague_unresolvable_followup`
* `reason`: Concrete justification for why the document does not answer the question

## Target Evaluation Metric

* **Primary Metric**: **Hit Rate@4** for answerable questions.
  * A query is marked as a **Hit** if at least one expected chunk appears within the **top 4 retrieved results**.
  * Unanswerable queries are reported separately and do not count as positive hits.
