# Agentix: Autonomous Multimodal Multi-Agent RAG System

Agentix is an enterprise-grade, multimodal, multi-agent Retrieval-Augmented Generation (RAG) platform. Orchestrated via **LangGraph**, Agentix features intent-based supervisor routing, conversational follow-up resolution, hybrid dense/sparse vector retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking with calibrated relevance thresholds, whole-document map-reduce summarization, and self-correcting reflection loops.

Built on an asynchronous **FastAPI** backend with **PostgreSQL (pgvector)** and a reactive **React 19** frontend, Agentix delivers real-time Server-Sent Events (SSE) token streaming, active agent telemetry, and citation verification over private document collections.

---

## Key Features

- **Autonomous Intent Routing & Query Rewriting:** A supervisor agent classifies user queries into five operational intents (`GREETING`, `DIRECT_ANSWER`, `RAG_QUERY`, `WEB_SEARCH`, `SUMMARIZE_DOCUMENT`) while resolving conversational pronouns, ordinals, and ambiguous follow-ups into self-contained standalone search queries using conversation history.
- **Multimodal Universal Ingestion:** Ingests and processes PDF (with PyMuPDF page rendering and Gemini Vision OCR fallback for scanned pages), Word (`.docx` with paragraph-table sequence and table markdown formatting), Excel (`.xlsx`, `.xls` with structured row serialization and sheet header retention), CSV, TXT, Markdown, JSON (with multi-encoding and BOM detection), and raw query images (PNG, JPEG, WEBP, GIF).
- **Hybrid Dense & Sparse Retrieval:** Combines dense cosine semantic search (`pgvector` with HNSW indexing) and sparse lexical search (PostgreSQL `to_tsvector` full-text search with GIN indexing) fused via Reciprocal Rank Fusion (RRF, $k=60$).
- **Cross-Encoder Reranking & Relevance Calibration:** Integrates `cross-encoder/ms-marco-MiniLM-L-6-v2` with empirical score thresholds ($0.30$ rerank score, $0.68$ dense cosine similarity) to suppress hallucinated answers on out-of-corpus queries.
- **Whole-Document Summarization Engine:** Deterministically detects whole-document summary commands, bypassing retrieval candidate limits to load all document chunks and execute single-call ($\le 32\text{k}$ chars) or map-reduce ($24\text{k}$ char chunks with 5-call budget caps) summarization with rate-limit protection.
- **Self-Correcting Reflection Loop:** A dedicated evaluation agent inspects draft responses for groundedness and relevance. If an answer fails, it feeds targeted feedback to the supervisor to rewrite the search query and re-execute retrieval (capped at 2 retries with an unverified fallback caveat).
- **Real-Time Token & Agent Streaming:** Delivers token deltas, active agent node status chips, source citations with page numbers, and verification badges to the React frontend via Server-Sent Events (SSE).
- **Scoped Tenant Isolation & Deduplication:** Multi-level data boundaries across Users, Knowledge Bases, and Chat Sessions enforced by database cascading foreign keys, SHA-256 content hash deduplication, and Supabase ES256 JWKS asymmetric token validation.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React 19 + Vite)"]
        UI["Chat Interface & Knowledge Base Manager"]
        SSEParser["SSE Stream Parser (parseSSEStream)"]
        AxiosClient["Axios Client (Auth Interceptor)"]
    end

    subgraph Gateway["API Gateway (FastAPI)"]
        AuthDep["Auth Middleware (JWKS ES256 / Supabase)"]
        ChatRouter["Chat Router (/chat, /chat/stream)"]
        UploadRouter["Upload Router (/upload, /knowledge-bases/.../documents)"]
        KBRouter["Knowledge Base Router (/knowledge-bases)"]
        SessionRouter["Session Router (/sessions)"]
        Recovery["Startup Recovery Service (recover_stuck_ingestion_jobs)"]
    end

    subgraph Storage["Persistence & Storage"]
        PG[("PostgreSQL 16 + pgvector")]
        S3Storage["Supabase Storage Bucket ('documents')"]
    end

    subgraph Pipeline["Multi-Agent Workflow (LangGraph)"]
        Supervisor["Supervisor Agent (Intent & Rewriter)"]
        VectorSearch["Vector Search Node (Dense + FTS + RRF)"]
        WebSearch["Web Search Tool (DuckDuckGo ddgs)"]
        Summarizer["Summarize Document Node (Map-Reduce)"]
        AnswerAgent["Answer Synthesis Agent"]
        ReflectionAgent["Reflection Evaluator Agent"]
        GreetingAgent["Greeting Agent"]
        DirectAgent["Direct Answer Agent"]
    end

    subgraph AIModels["External AI & Local Models"]
        GeminiFlash["Google Gemini 3.5 Flash (LLM)"]
        GeminiEmbed["Google Gemini Embedding 2 (2000 Dims)"]
        GeminiVision["Gemini Vision (OCR & Query Extraction)"]
        CrossEncoder["Local Cross-Encoder (ms-marco-MiniLM-L-6-v2)"]
    end

    UI -->|HTTP / REST| AxiosClient
    UI -->|SSE Stream| SSEParser
    AxiosClient --> AuthDep
    SSEParser --> AuthDep
    AuthDep --> ChatRouter & UploadRouter & KBRouter & SessionRouter
    Recovery -.->|Startup Sanity| PG

    UploadRouter -->|Upload File Bytes| S3Storage
    UploadRouter -->|Metadata & Status| PG
    UploadRouter -->|Background Ingestion| GeminiVision & GeminiEmbed & PG

    ChatRouter --> Pipeline
    Pipeline --> GeminiFlash
    VectorSearch --> PG
    VectorSearch --> GeminiEmbed
    VectorSearch --> CrossEncoder
    WebSearch -->|Web Query| AIModels
```

---

## Agentic RAG Workflow

The agentic pipeline is built on **LangGraph** (`StateGraph`) using a shared, typed execution state (`AgentState`).

```mermaid
flowchart TD
    START([User Query Received]) --> supervisor["Supervisor Agent<br/>(Intent Classifier & Query Rewriter)"]

    supervisor -->|intent = GREETING| greeting["Greeting Agent"]
    supervisor -->|intent = DIRECT_ANSWER| direct_answer["Direct Answer Agent"]
    supervisor -->|intent = SUMMARIZE_DOCUMENT| summarize["Summarize Document Node<br/>(Map-Reduce Engine)"]
    supervisor -->|intent = WEB_SEARCH| web_search["Web Search Node<br/>(DuckDuckGo ddgs)"]
    supervisor -->|intent = RAG_QUERY| vector_search["Vector Search Node<br/>(Hybrid HNSW + FTS + RRF)"]
    supervisor -->|intent = STOP_RETRY| END_NODE([Final Response])

    vector_search -->|Candidates >= Threshold| answer["Answer Synthesis Agent"]
    vector_search -->|0 Chunks in KB Chat & allow_web_search=True| web_search
    vector_search -->|0 Chunks in KB Chat & allow_web_search=False| answer
    vector_search -->|0 Chunks in Non-KB Chat| web_search
    vector_search -->|Retrieval Error| answer

    web_search --> answer

    answer -->|SKIP_REFLECTION=True| END_NODE
    answer -->|Intent in GREETING, DIRECT_ANSWER, WEB_SEARCH, SUMMARIZE| END_NODE
    answer -->|Not Covered / 0 Chunks| END_NODE
    answer -->|RAG Query with Document Context| reflection["Reflection Evaluator Node<br/>(Groundedness & Relevance)"]

    reflection -->|Verdict: Passed| END_NODE
    reflection -->|Verdict: Failed & retry_count < 2| supervisor
    reflection -->|Verdict: Failed & retry_count >= 2| END_NODE

    greeting --> END_NODE
    direct_answer --> END_NODE
    summarize --> END_NODE
```

### Agent Responsibilities

#### 1. Supervisor Agent (`ai/agents/supervisor.py`)
- **Intent Categorization:** Uses structured LLM outputs (`IntentClassification`) to categorize queries into `GREETING`, `DIRECT_ANSWER`, `RAG_QUERY`, `WEB_SEARCH`, or `SUMMARIZE_DOCUMENT`.
- **Conversational Context Rewriting:** Resolves ambiguous pronouns (`it`, `that`), ordinals (`the second one`), and follow-ups (`what about Q3?`) into self-contained standalone search queries using the prior 6 conversation messages.
- **Fast-Path Heuristics:**
  - *Greeting Fast-Path:* Detects standalone greetings (`hi`, `hello`, `hey`) without history or images, bypassing LLM classification.
  - *Web Search Confirmation:* Recognizes user consent phrases (`yes please`, `sure`, `search online`) following an assistant "not covered" message, resolving back to the original question.
  - *Document Summary Detection:* Matches whole-document summary phrases via regex to route directly to summarization.
- **Query Rewrite on Retry:** On reflection failure, takes evaluator critique and previous queries, formulating a novel targeted search query via `RetryQueryRewrite` (with deterministic fallback `build_deterministic_retry_query`).

#### 2. Greeting Agent (`ai/agents/greeting.py`)
- Handles conversational pleasantries, introductions, and capabilities questions without performing tool lookups or database queries.

#### 3. Direct Answer Agent (`ai/agents/direct_answer.py`)
- Answers general knowledge, mathematics, programming, and reasoning queries that require no external document lookup.

#### 4. Summarize Document Node (`ai/services/summarization_service.py`)
- Resolves the target document within user and session/KB scope.
- Bypasses similarity thresholds to fetch all document chunks ordered by `chunk_index`.
- Executes single-call synthesis for documents $\le 32,000$ characters, or multi-batch map-reduce for documents $> 32,000$ characters with rate-limit protections (maximum 4 map calls + 1 reduce call).

#### 5. Vector Search Node (`ai/tools/vector_search.py` & `ai/services/retrieval_service.py`)
- Queries PostgreSQL for dense semantic matches (pgvector HNSW) and sparse keyword matches (PostgreSQL FTS), performs Reciprocal Rank Fusion, applies cross-encoder reranking, and filters out candidates failing the calibrated relevance threshold.

#### 6. Web Search Node (`ai/tools/web_search.py`)
- Queries DuckDuckGo via `ddgs` inside a thread pool (`asyncio.to_thread`), returning source titles, URLs, and text snippets formatted into context blocks and UI citation schemas.

#### 7. Answer Synthesis Agent (`ai/agents/answer.py`)
- Synthesizes factual answers strictly grounded in retrieved context chunks.
- Dynamically selects between `ANSWER_SYSTEM_PROMPT` (document RAG) and `WEB_ANSWER_SYSTEM_PROMPT` (live web).
- Returns the standard "not covered" message when 0 chunks pass the relevance threshold:  
  `"I couldn't find this in your uploaded documents. Would you like me to search the web instead?"`

#### 8. Reflection Evaluator Agent (`ai/agents/reflection.py`)
- Evaluates draft answers against context and query using structured output `ReflectionOutput`:
  - `is_grounded`: Factual verification against context.
  - `is_relevant`: Direct relevance to user question.
- Triggers retry loops if an evaluation fails and retry count is below 2. If the retry cap is reached, appends an unverified warning caveat to the output:  
  `> ⚠️ **Note:** This response could not be fully verified against your uploaded documents and may contain incomplete or ungrounded details.`

---

## Document Ingestion Pipeline

```mermaid
flowchart TD
    A[Client File Upload] --> B{Validation}
    B -->|Size > 10MB or Empty| C[Reject HTTP 400 / 413]
    B -->|Valid Bytes| D[Compute SHA-256 content_hash]

    D --> E{Scope Deduplication Check<br/>user_id + scope + hash}
    E -->|Duplicate Exists| F[Return HTTP 200 with existing doc ID]
    E -->|Novel Document| G[Upload to Supabase Storage<br/>path: user_id/scope/uuid_filename]

    G --> H[Insert Document Record<br/>status='processing']
    H --> I[Dispatch Background Task<br/>process_document_background]
    I --> J[Return HTTP 202 ACCEPTED]

    subgraph Background Execution
        J -.-> K[Parser Service: parse_document]
        K --> L{Format Parser}
        L -->|PDF| M[PyMuPDF text extraction<br/>+ Gemini Vision OCR for scans]
        L -->|DOCX| N[python-docx: body iteration<br/>preserving paragraphs & tables]
        L -->|Excel XLSX/XLS| O[pandas: per-sheet structured rows<br/>col1: val1 | col2: val2]
        L -->|CSV| P[pandas + charset-normalizer]
        L -->|Images| Q[Gemini Vision describe_image]
        L -->|Text/MD/JSON| R[decode_text_bytes BOM & charset detection]

        M & N & O & P & Q & R --> S[Chunking Service: chunk_text<br/>chunk_size=600, overlap=120]
        S --> T[Embedding Service: generate_batch_embeddings<br/>gemini-embedding-2, 2000 dims, batches of 32]
        T --> U[Insert DocumentChunk records<br/>embedding + computed fts_tokens]
        U --> V[Verify Chunk Count Matches Database]
        V --> W[Update Document status='ready']
        W -.->|On Failure| X[Update Document status='failed'<br/>record error_message]
    end
```

### Parsing Specifications
- **PDF Documents:** Evaluates native text per page with `pypdf` and PyMuPDF (`fitz`). If a page has sparse text ($< 50$ characters) and contains images, or if text quality is corrupted (printable character ratio $< 70\%$), it renders the page at 2x scale (144 DPI) and triggers Gemini Vision OCR.
- **Word Documents (`.docx`):** Traverses document body elements in XML sequence, maintaining relative ordering of paragraphs and tables. Converts tables into Markdown format, handles merged cells, and extracts sections' headers and footers. Rejects legacy binary `.doc` files with an explanatory error.
- **Spreadsheets (`.xlsx`, `.xls`):** Parses all sheets with `openpyxl`/`xlrd` into pandas DataFrames. Serializes rows into structured records (`col1: val1 | col2: val2`), drops phantom empty columns, and prefixes every chunk with `Sheet: <sheet_name>`.
- **Delimited Files (`.csv`):** Decodes raw bytes with `charset-normalizer`, parses via pandas, and formats rows with header names.
- **Plain Text / Code (`.txt`, `.md`, `.json`):** Inspects Byte Order Marks (UTF-32, UTF-8, UTF-16), detects null-byte patterns, and normalizes encodings with Western codepage priority (CP1252/Latin-1).
- **Standalone Images (`.png`, `.jpg`, `.jpeg`, `.webp`):** Validates magic bytes, enforces 10MB limits, and sends image bytes to Gemini Vision to produce descriptions for indexing.

### Chunking & Embedding Specifications
- **Chunk Size & Overlap:** `chunk_size = 600` characters, `chunk_overlap = 120` characters. Recursively splits along semantic boundaries (`\n\n`, `\n`, `. `, `? `, `! `, ` `, `""`).
- **Header Propagation:** Preserves table and sheet headers by prepending them to every generated child chunk.
- **Embedding Ingestion:** Embeds chunks in batches of 32 using `gemini-embedding-2` with prefix `title: {filename} | text: {chunk}`. Output dimensions are truncated to 2000 via API configuration to stay within pgvector's HNSW index limits.
- **Interrupted Job Recovery:** During startup, `recover_stuck_ingestion_jobs()` scans for documents in `processing` status older than 300 seconds, sets their status to `failed`, and deletes orphaned chunks.

---

## Retrieval Architecture

### 1. Hybrid Search (Dense + Sparse)
Retrieval operates across two query representations:
1. **Dense Retrieval:** Generates a 2000-dimensional vector of the query (prefixed with `task: search result | query: {query}`). Executes cosine distance search (`dc.embedding <=> CAST(:query_vector AS vector)`) against the HNSW index (`m=16`, `ef_construction=64`).
2. **Sparse Retrieval:** Formats query words into a disjunctive websearch query (`term1 or term2`) executed against PostgreSQL's computed TSVector column (`dc.fts_tokens`) using a GIN index and `ts_rank_cd`.

### 2. Reciprocal Rank Fusion (RRF)
Top 30 candidates from each pipeline are merged using Reciprocal Rank Fusion with smoothing constant $k = 60$:

$$\text{RRF Score} = \frac{1}{\text{rank}_{\text{sparse}} + 60} + \frac{1}{\text{rank}_{\text{dense}} + 60}$$

### 3. Cross-Encoder Reranking
Top 20 candidates from RRF are scored by a local `cross-encoder/ms-marco-MiniLM-L-6-v2` model:
- Accepts pairs of `[query, chunk_content]`.
- Computes raw logits and applies sigmoid activation: $\text{score} = \sigma(\text{logit})$.
- Pre-warmed during application startup to eliminate cold-start inference latency.

### 4. Relevance Threshold Filtering
Surviving candidates are filtered against calibrated score cutoffs:
- **Reranker Enabled (Default):** Candidates must meet `MIN_RELEVANCE_RERANK = 0.30`.
- **Reranker Disabled (`RERANK_PROVIDER="none"`):** Candidates must meet `MIN_RELEVANCE_DENSE = 0.68` (cosine similarity).
- Chunks falling below the cutoff are dropped. The top 4 surviving chunks are passed to the answer agent. If 0 chunks survive, the system triggers the "not covered" flow.

---

## Database Architecture

```mermaid
erDiagram
    users ||--o{ chat_sessions : owns
    users ||--o{ knowledge_bases : owns
    users ||--o{ documents : owns
    knowledge_bases ||--o{ documents : contains
    knowledge_bases ||--o{ chat_sessions : scopes
    chat_sessions ||--o{ messages : contains
    chat_sessions ||--o{ documents : attaches
    chat_sessions ||--o{ agent_logs : records
    documents ||--o{ document_chunks : splits_into

    users {
        uuid id PK
        string email UK
        string full_name
        string avatar_url
        timestamptz created_at
        timestamptz updated_at
    }

    knowledge_bases {
        uuid id PK
        uuid user_id FK
        string name
        text description
        timestamptz created_at
        timestamptz updated_at
    }

    chat_sessions {
        uuid id PK
        uuid user_id FK
        uuid knowledge_base_id FK
        string title
        timestamptz created_at
        timestamptz updated_at
    }

    messages {
        uuid id PK
        uuid session_id FK
        string sender
        text content
        jsonb citations
        timestamptz created_at
    }

    documents {
        uuid id PK
        uuid user_id FK
        uuid session_id FK
        uuid knowledge_base_id FK
        string filename
        string file_type
        string file_path
        integer file_size
        string status
        text error_message
        integer ocr_warning_count
        string content_hash
        timestamptz created_at
        timestamptz updated_at
    }

    document_chunks {
        uuid id PK
        uuid document_id FK
        integer chunk_index
        integer page_number
        text content
        jsonb metadata
        vector embedding "Vector(2000)"
        tsvector fts_tokens "Computed"
        timestamptz created_at
    }

    agent_logs {
        uuid id PK
        uuid session_id FK
        string agent_name
        jsonb input_data
        jsonb output_data
        float execution_time_ms
        timestamptz created_at
    }
```

### Table Definitions & Key Indexes
- **`users`:** Stores local synchronized user profiles. Primary key `id` (UUID) maps to Supabase Auth UID.
- **`knowledge_bases`:** Named collections of documents. Foreign key `user_id` cascades on delete.
- **`chat_sessions`:** Conversation sessions. Foreign key `knowledge_base_id` sets null on delete; `user_id` cascades on delete.
- **`messages`:** Chat history. Stores `sender` (`user` or `assistant`), text `content`, and JSONB `citations`.
- **`documents`:** Stores upload metadata, Supabase storage path, parsing status (`processing`, `ready`, `failed`), and SHA-256 `content_hash`. Contains three scoped partial unique indexes for deduplication:
  - `uq_documents_session_content_hash`: `(user_id, session_id, content_hash)`
  - `uq_documents_kb_content_hash`: `(user_id, knowledge_base_id, content_hash)`
  - `uq_documents_global_content_hash`: `(user_id, content_hash)`
- **`document_chunks`:** Stores chunked text and retrieval vectors:
  - `embedding`: `Vector(2000)` indexed with HNSW (`vector_cosine_ops`, `m=16`, `ef_construction=64`).
  - `fts_tokens`: Generated stored column (`to_tsvector('english', content)`) indexed with GIN (`idx_document_chunks_fts`).
- **`agent_logs`:** Audit table capturing agent execution time, node names, inputs, and outputs.

---

## Technology Stack Reference

| Layer | Component | Version / Model | Role in Agentix |
| :--- | :--- | :--- | :--- |
| **Backend** | Python | `3.13` (tested), $\ge 3.11$ | Backend language runtime |
| | FastAPI | $\ge 0.115.0$ | Core web framework, dependency injection, async routes |
| | Uvicorn | $\ge 0.30.0$ | ASGI server with async loop |
| | Pydantic / Pydantic Settings | $\ge 2.7.0$ / $\ge 2.3.0$ | Data contracts, validation, settings management |
| **Database** | PostgreSQL | `16` | Relational store and vector engine |
| | pgvector | $\ge 0.3.0$ | HNSW cosine vector index (`Vector(2000)`) |
| | SQLAlchemy | $\ge 2.0.0$ | Async ORM and connection pooling |
| | asyncpg | $\ge 0.29.0$ | Async driver for application runtime |
| | psycopg2-binary | $\ge 2.9.0$ | Synchronous driver for Alembic migrations |
| | Alembic | $\ge 1.13.0$ | Database migration management |
| **AI & LLM** | LangGraph | $\ge 0.2.0$ | Cyclical multi-agent orchestration |
| | LangChain Core / Google GenAI | $\ge 0.3.0$ / $\ge 2.0.0$ | `ChatGoogleGenerativeAI` wrapper |
| | Google GenAI SDK | $\ge 1.0.0$ | Client for `gemini-embedding-2` |
| | Google Gemini 3.5 Flash | API (`gemini-3.5-flash`) | Intent routing, response generation, query rewriting |
| | Google Gemini Embedding 2 | API (`gemini-embedding-2`)| 2000-dimensional embeddings |
| | Sentence-Transformers | $\ge 6.0.1$ | `cross-encoder/ms-marco-MiniLM-L-6-v2` local reranker |
| | PyTorch | $\ge 2.14.1$ (CPU) | Reranker tensor execution and dynamic quantization |
| **Tools & Parsers**| DuckDuckGo Search | `ddgs` $\ge 9.0.0$ | DuckDuckGo web search |
| | pypdf & PyMuPDF (fitz) | $\ge 4.0.0$ / dynamic | Native PDF parsing & 2x scale page rendering |
| | python-docx | $\ge 1.1.0$ | Word document parsing & table markdown formatting |
| | pandas, openpyxl, xlrd | $\ge 2.2.0$, $\ge 3.1.5$, $\ge 2.0.2$| Excel and CSV structured row serialization |
| | charset-normalizer | $\ge 3.3.0$ | Text encoding and BOM detection |
| **Frontend** | React | `19.2.8` | Declarative UI components and hooks |
| | Vite | `8.2.0` | Frontend dev server and bundler |
| | Tailwind CSS | `4.3.3` | Utility-first styling with dark mode support |
| | React Router DOM | `7.18.2` | Client-side routing and protected routes |
| | Supabase JS | `^2.112.2` | Supabase auth client with tab-scoped session storage |
| | Axios | `^1.19.0` | HTTP client with automatic JWT bearer injection |
| | Lucide React | `^1.30.0` | UI icon components |

---

## Project Directory Structure

```
Agentix/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI workflow: PostgreSQL pgvector, Alembic, Pytest, Node lint & build
├── backend/
│   ├── ai/
│   │   ├── agents/               # LangGraph agent implementations
│   │   │   ├── answer.py         # Grounded answer synthesis agent
│   │   │   ├── direct_answer.py  # Standalone general knowledge agent
│   │   │   ├── graph.py          # StateGraph assembly and conditional routing edges
│   │   │   ├── greeting.py       # Greeting fast-path agent
│   │   │   ├── reflection.py     # Groundedness & relevance evaluation agent
│   │   │   ├── state.py          # AgentState TypedDict definition
│   │   │   └── supervisor.py     # Intent classification & retry query rewriter
│   │   ├── prompts/              # Agent system prompts
│   │   │   ├── answer_prompt.py
│   │   │   ├── direct_answer_prompt.py
│   │   │   ├── greeting_prompt.py
│   │   │   └── supervisor_prompt.py
│   │   ├── services/             # Core AI, ingestion, and retrieval services
│   │   │   ├── chunking_service.py      # Semantic chunking (600/120) with header propagation
│   │   │   ├── embedding_service.py     # Gemini embeddings & CrossEncoder reranker
│   │   │   ├── llm_service.py           # ChatGoogleGenerativeAI client & image descriptor
│   │   │   ├── parser_service.py        # Universal document parser (PDF, Word, Excel, CSV, Text)
│   │   │   ├── retrieval_service.py     # Hybrid search (Dense HNSW + Sparse FTS) + RRF + Reranking
│   │   │   ├── summarization_service.py # Whole-document single-call & map-reduce summarization
│   │   │   └── vision_service.py        # Query image validation & retrieval term extraction
│   │   └── tools/                # Specialized agent retrieval tools
│   │       ├── vector_search.py  # Scoped hybrid search tool adapter
│   │       └── web_search.py     # Asynchronous DuckDuckGo search tool
│   ├── alembic/                  # Database migration scripts
│   ├── app/                      # Application backend source code
│   │   ├── auth/
│   │   │   └── dependencies.py   # JWT ES256 JWKS verification, remote fallback, user sync
│   │   ├── config/
│   │   │   └── settings.py       # Pydantic BaseSettings & driver rewrites
│   │   ├── database/
│   │   │   ├── connection.py     # Async database engine, pool settings, get_db
│   │   │   ├── crud.py           # Database CRUD operations
│   │   │   └── queries.py        # (Queries centralized in retrieval_service.py)
│   │   ├── models/               # SQLAlchemy Declarative Models
│   │   ├── routes/               # FastAPI route controllers
│   │   ├── schemas/              # Pydantic validation schemas
│   │   ├── services/
│   │   │   └── recovery.py       # Ingestion failure recovery on startup
│   │   └── main.py               # FastAPI factory, lifespan, CORS, and router registration
│   ├── evaluation/               # Benchmark datasets and evaluation reports
│   ├── scripts/                  # Evaluation runner scripts
│   ├── tests/                    # 38 unit and integration test files
│   ├── alembic.ini               # Alembic configuration
│   ├── pytest.ini                # Pytest configuration
│   └── requirements.txt          # Python dependencies
└── frontend/
    ├── public/                   # Static logos, icons, and favicons
    ├── src/
    │   ├── api/                  # Axios, Supabase client, and REST abstractions
    │   ├── components/           # UI components (Chat, KnowledgeBase, Sidebar, Upload)
    │   ├── context/              # Context providers (Auth, Chat, KnowledgeBase, Theme)
    │   ├── hooks/                # Custom hooks (useAuth, useChat, useUpload, etc.)
    │   ├── layouts/              # AuthLayout, ChatLayout
    │   ├── pages/                # Login, Register, ChatPage, KBListPage, KBDetailPage, NotFound
    │   ├── routes/               # AppRoutes route definitions
    │   ├── utils/                # streamParser, chatHelpers, constants, helpers
    │   ├── App.jsx               # Provider tree root
    │   ├── index.css             # Tailwind CSS imports
    │   └── main.jsx              # DOM entry point
    ├── package.json              # NPM dependencies and scripts
    └── vite.config.js            # Vite build configuration
```

---

## API Inventory

All API endpoints are mounted under `/api/v1` (except the root welcome endpoint):

| Method | Route | Description | Auth Required | Request Payload | Response Schema |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `GET` | `/` | Root API greeting | No | None | JSON greeting |
| `GET` | `/api/v1/health` | System health check | No | None | `HealthStatus` (200 OK) |
| `POST` | `/api/v1/chat` | Synchronous chat execution | Yes | `ChatRequest` (JSON) | `ChatResponse` (200 OK) |
| `POST` | `/api/v1/chat/stream` | Server-Sent Events token stream | Yes | `ChatRequest` (JSON) | `text/event-stream` (200 OK) |
| `POST` | `/api/v1/sessions` | Create a new conversation session | Yes | `SessionCreate` (JSON) | `SessionResponse` (201 Created) |
| `GET` | `/api/v1/sessions` | List caller's chat sessions | Yes | Query: `limit`, `offset` | `List[SessionResponse]` (200 OK) |
| `GET` | `/api/v1/sessions/{id}` | Get session metadata by ID | Yes | Path: `session_id` (UUID) | `SessionResponse` (200 OK) |
| `PATCH` | `/api/v1/sessions/{id}` | Rename chat session title | Yes | `SessionUpdate` (JSON) | `SessionResponse` (200 OK) |
| `DELETE` | `/api/v1/sessions/{id}` | Delete chat session and cascade | Yes | Path: `session_id` (UUID) | Empty (204 No Content) |
| `GET` | `/api/v1/sessions/{id}/messages` | Get session message history | Yes | Query: `limit`, `offset` | `List[MessageResponse]` (200 OK) |
| `POST` | `/api/v1/knowledge-bases` | Create a new Knowledge Base | Yes | `KnowledgeBaseCreate` (JSON)| `KnowledgeBaseResponse` (201 Created) |
| `GET` | `/api/v1/knowledge-bases` | List caller's Knowledge Bases | Yes | Query: `limit`, `offset` | `List[KnowledgeBaseResponse]` (200 OK) |
| `GET` | `/api/v1/knowledge-bases/{id}` | Get Knowledge Base by ID | Yes | Path: `kb_id` (UUID) | `KnowledgeBaseResponse` (200 OK) |
| `PATCH` | `/api/v1/knowledge-bases/{id}` | Update Knowledge Base | Yes | `KnowledgeBaseUpdate` (JSON)| `KnowledgeBaseResponse` (200 OK) |
| `DELETE` | `/api/v1/knowledge-bases/{id}` | Delete Knowledge Base | Yes | Path: `kb_id` (UUID) | Empty (204 No Content) |
| `GET` | `/api/v1/knowledge-bases/{id}/documents` | List documents in Knowledge Base | Yes | Query: `limit`, `offset` | `DocumentListResponse` (200 OK) |
| `POST` | `/api/v1/knowledge-bases/{id}/documents` | Upload document to Knowledge Base| Yes | Multipart form (`file`) | `DocumentUploadResponse` (202 / 200) |
| `DELETE` | `/api/v1/knowledge-bases/{id}/documents/{doc_id}` | Delete document from KB | Yes | Path: `kb_id`, `document_id` | Empty (204 No Content) |
| `POST` | `/api/v1/upload` | Upload document to chat session | Yes | Multipart form (`session_id`, `file`) | `DocumentUploadResponse` (202 / 200) |
| `POST` | `/api/v1/search` | Direct vector search endpoint | Yes | `SearchRequest` (JSON) | `SearchResponse` (200 OK) |

---

## Environment Configuration

### Backend (`backend/.env`)

| Variable Name | Type | Required | Default Value | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| `DATABASE_URL` | `str` | **Yes** | `""` | PostgreSQL connection URL (e.g. `postgresql+asyncpg://user:pass@host:5432/db`) |
| `SUPABASE_URL` | `str` | **Yes** | `""` | Supabase project URL (used for JWKS token verification and storage) |
| `SUPABASE_KEY` | `str` | **Yes** | `""` | Supabase service key (used for storage file uploads) |
| `GEMINI_API_KEY` | `str` | **Yes** | `""` | Google Gemini API key for embeddings and generation |
| `LLM_MODEL` | `str` | No | `"gemini-3.5-flash"` | Gemini model used for agent reasoning and answer synthesis |
| `GEMINI_EMBEDDING_MODEL`| `str` | No | `"gemini-embedding-2"` | Model used for dense text embedding |
| `EMBEDDING_DIMENSIONS` | `int` | No | `2000` | Vector dimensionality truncated via API to respect HNSW cap |
| `EMBED_CONCURRENCY` | `int` | No | `4` | Semaphore concurrency for batch embedding generation |
| `MAX_UPLOAD_SIZE_MB` | `int` | No | `10` | Maximum file upload size in megabytes |
| `SKIP_REFLECTION` | `bool`| No | `False` | Setting to `True` bypasses the reflection evaluation node |
| `RERANK_PROVIDER` | `str` | No | `"local"` | Set to `"local"` for CrossEncoder, or `"none"` to disable |
| `RERANK_MODEL_NAME` | `str` | No | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace cross-encoder model identifier |
| `MIN_RELEVANCE_RERANK` | `float`| No | `0.30` | Cross-encoder minimum score threshold |
| `MIN_RELEVANCE_DENSE` | `float`| No | `0.68` | Cosine similarity threshold when reranker is disabled |
| `HF_TOKEN` | `str` | No | `""` | Optional HuggingFace token for gated models |

### Frontend (`frontend/.env`)

| Variable Name | Type | Required | Default Value | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| `VITE_SUPABASE_URL` | `str` | **Yes** | `""` | Supabase project URL for frontend client authentication |
| `VITE_SUPABASE_ANON_KEY` | `str` | **Yes** | `""` | Supabase anonymous public key |
| `VITE_API_BASE_URL` | `str` | No | `"http://127.0.0.1:8000/api/v1"` | URL pointing to the FastAPI backend API prefix |

---

## Installation & Setup

### Prerequisites
- Python 3.11+ (Python 3.13 recommended)
- Node.js 20+ and npm
- PostgreSQL 16 with `pgvector` extension installed (`CREATE EXTENSION IF NOT EXISTS vector;`)
- Supabase account with an active project and a public storage bucket named `documents`
- Google Gemini API key

### 1. Database Setup
Ensure PostgreSQL is running and initialize the vector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Backend Installation & Startup
```bash
# Navigate to the backend directory
cd backend

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Create your configuration file: backend/.env
# Example content:
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-supabase-service-role-key
# GEMINI_API_KEY=your-google-gemini-api-key
# LLM_MODEL=gemini-3.5-flash
# RERANK_PROVIDER=local

# Run database migrations with Alembic
alembic upgrade head

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API documentation is accessible at `http://127.0.0.1:8000/docs`.

### 3. Frontend Installation & Startup
```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Create your frontend configuration file: frontend/.env
# Example content:
# VITE_SUPABASE_URL=https://your-project.supabase.co
# VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
# VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1

# Start the Vite development server
npm run dev
```
The web application is accessible at `http://localhost:5173`.

---

## Testing & Evaluation

### Backend Test Suite
The repository includes **38 test files** validating unit, integration, and security logic:
```bash
cd backend
pytest
```

Key test modules:
- `tests/test_routing.py` & `tests/test_followup_query.py`: Intent classification, conversational pronouns, and query contextualization.
- `tests/test_retrieval.py` & `tests/test_relevance_threshold.py`: Dense/sparse hybrid search, RRF scoring, and threshold filtering.
- `tests/test_parser_service.py` & `tests/test_docx_parsing.py`: File format parsers, table formatting, and encoding checks.
- `tests/test_reflection.py` & `tests/test_reflection_max_retry.py`: Self-correcting reflection loops and retry limits.
- `tests/test_summarization.py`: Whole-document single-call and map-reduce summarization logic.
- `tests/test_deduplication.py`: SHA-256 content deduplication and scoped unique constraints.
- `tests/test_recovery.py`: Interrupted ingestion job recovery.

### Retrieval Benchmark Suite
To evaluate retrieval effectiveness (Hit Rate@4, MRR@4, Binary nDCG@4) on the benchmark corpus:
```bash
cd backend
python scripts/eval_retrieval.py --dataset evaluation/retrieval_questions_v2.json --output-dir evaluation/results/expanded_corpus_v2
```

---

## License

This project is licensed under the MIT License.
