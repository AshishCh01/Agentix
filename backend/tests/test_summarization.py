import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.agents.graph import (
    build_graph,
    route_after_answer,
    route_intent,
    summarize_document_node,
)
from ai.agents.supervisor import IntentClassification, classify_intent
from ai.services.summarization_service import (
    BATCH_CHAR_TARGET,
    MAX_MAP_CALLS,
    SINGLE_CALL_CHAR_BUDGET,
    RateLimitExceededError,
    generate_document_summary,
    is_summarize_document_query,
    resolve_target_document,
)
from app.models.document import Document
from app.models.document_chunks import DocumentChunk


# =====================================================================
# 1. Deterministic Query Detection Tests
# =====================================================================

@pytest.mark.parametrize(
    "query",
    [
        "summarize this document",
        "summarize the document",
        "can you summarize this document?",
        "please summarize this file",
        "give me a summary of this document",
        "give me an overview of this pdf",
        "tl;dr of this document",
        "tldr of the document",
        "give me the gist of this document",
        "summarize my document",
        "summarize the document I just uploaded",
        "what I just uploaded, summarize it",
        "summarize handbook.pdf",
        "give me an overview of report.docx",
        "tl;dr of notes.txt",
        "summarize the entire document",
        "summarize the whole document",
        "give me a summary of everything in this document",
        "summarize",
        "tl;dr",
        "give me a summary",
        "can you summarize it?",
    ],
)
def test_is_summarize_document_query_positive(query: str):
    assert is_summarize_document_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "summarize the section on vacation policy",
        "summarize what it says about pricing",
        "summarize the chapter regarding authentication",
        "summary of the paragraph about warranty",
        "give me a summary of the policy on refunds",
        "what does this document say about security?",
        "how to configure ssl in this document?",
        "why is the system slow in this document?",
        "who is responsible for data retention?",
        "what is the capital of France?",
        "write a python function to reverse a string",
        "hello there",
        "What is machine learning?",
    ],
)
def test_is_summarize_document_query_negative(query: str):
    assert is_summarize_document_query(query) is False


# =====================================================================
# 2. Supervisor Classification Tests
# =====================================================================

@pytest.mark.asyncio
async def test_supervisor_summarize_fast_path():
    """Whole-document summarization triggers deterministic fast-path without LLM calls."""
    state = {"user_query": "can you summarize this document?"}
    res = await classify_intent(state)
    assert res["intent"] == "SUMMARIZE_DOCUMENT"
    assert res["standalone_query"] == "can you summarize this document?"


@pytest.mark.asyncio
async def test_supervisor_topic_focused_routes_to_rag():
    """Topic-focused queries do not trigger fast-path and route to RAG_QUERY."""
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.ainvoke.return_value = IntentClassification(
        intent="RAG_QUERY",
        standalone_query="summarize the section on refund policy in the document",
    )

    with patch("ai.agents.supervisor.llm_service.get_chat_model", return_value=mock_llm):
        state = {"user_query": "summarize the section on refund policy"}
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"


@pytest.mark.asyncio
async def test_supervisor_structured_output_failure_fallback():
    """If structured output throws an exception, fallback still identifies SUMMARIZE_DOCUMENT."""
    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=RuntimeError("API down")):
        state = {"user_query": "summarize handbook.pdf"}
        res = await classify_intent(state)
        assert res["intent"] == "SUMMARIZE_DOCUMENT"


# =====================================================================
# 3. Target Document Resolution & Scoping Tests
# =====================================================================

def make_doc(
    doc_id: uuid.UUID,
    user_id: uuid.UUID,
    filename: str,
    status: str = "ready",
    session_id: uuid.UUID = None,
    knowledge_base_id: uuid.UUID = None,
    error_message: str = None,
) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.user_id = user_id
    doc.filename = filename
    doc.status = status
    doc.session_id = session_id
    doc.knowledge_base_id = knowledge_base_id
    doc.error_message = error_message
    return doc


@pytest.mark.asyncio
async def test_resolve_target_document_named_match():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc1 = make_doc(uuid.uuid4(), user_id, "employee_handbook.pdf", session_id=session_id)
    doc2 = make_doc(uuid.uuid4(), user_id, "financial_report.pdf", session_id=session_id)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [doc1, doc2]
    mock_db.execute.return_value = mock_result

    target, status, _ = await resolve_target_document(
        db=mock_db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=None,
        query="summarize financial_report.pdf",
    )
    assert status is None
    assert target == doc2


@pytest.mark.asyncio
async def test_resolve_target_document_single_session_upload():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    # 2 KB documents, 1 session upload
    kb_doc1 = make_doc(uuid.uuid4(), user_id, "kb1.pdf", knowledge_base_id=kb_id)
    kb_doc2 = make_doc(uuid.uuid4(), user_id, "kb2.pdf", knowledge_base_id=kb_id)
    session_doc = make_doc(uuid.uuid4(), user_id, "upload.pdf", session_id=session_id, knowledge_base_id=None)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [session_doc, kb_doc1, kb_doc2]
    mock_db.execute.return_value = mock_result

    target, status, _ = await resolve_target_document(
        db=mock_db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=kb_id,
        query="summarize this document",
    )
    # Exactly one session upload -> picked automatically
    assert status is None
    assert target == session_doc


@pytest.mark.asyncio
async def test_resolve_target_document_just_uploaded():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_old = make_doc(uuid.uuid4(), user_id, "old.pdf", session_id=session_id)
    doc_new = make_doc(uuid.uuid4(), user_id, "new.pdf", session_id=session_id)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    # Ordered by created_at desc, so doc_new is first
    mock_result.scalars.return_value.all.return_value = [doc_new, doc_old]
    mock_db.execute.return_value = mock_result

    target, status, _ = await resolve_target_document(
        db=mock_db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=None,
        query="summarize the document I just uploaded",
    )
    assert status is None
    assert target == doc_new


@pytest.mark.asyncio
async def test_resolve_target_document_multiple_candidates_clarification():
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    doc1 = make_doc(uuid.uuid4(), user_id, "policy_a.pdf", knowledge_base_id=kb_id)
    doc2 = make_doc(uuid.uuid4(), user_id, "policy_b.pdf", knowledge_base_id=kb_id)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [doc1, doc2]
    mock_db.execute.return_value = mock_result

    target, status, candidates = await resolve_target_document(
        db=mock_db,
        user_id=user_id,
        session_id=None,
        knowledge_base_id=kb_id,
        query="summarize this document",
    )
    assert target is None
    assert status == "clarification_needed"
    assert len(candidates) == 2


@pytest.mark.asyncio
async def test_resolve_target_document_no_documents():
    user_id = uuid.uuid4()
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    target, status, candidates = await resolve_target_document(
        db=mock_db,
        user_id=user_id,
        session_id=uuid.uuid4(),
        knowledge_base_id=None,
        query="summarize this document",
    )
    assert target is None
    assert status == "no_documents"
    assert candidates == []


# =====================================================================
# 4. Status Handling & Scoping Protection Tests
# =====================================================================

@pytest.mark.asyncio
async def test_generate_summary_processing_status():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    proc_doc = make_doc(uuid.uuid4(), user_id, "processing_doc.pdf", status="processing", session_id=session_id)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [proc_doc]
    mock_db.execute.return_value = mock_result

    res = await generate_document_summary(
        db=mock_db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=None,
        query="summarize this document",
    )
    assert "still being processed" in res["final_response"]
    assert "not covered" not in res["final_response"].lower()
    assert "couldn't find this" not in res["final_response"].lower()


@pytest.mark.asyncio
async def test_generate_summary_failed_status():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    failed_doc = make_doc(
        uuid.uuid4(),
        user_id,
        "broken.pdf",
        status="failed",
        session_id=session_id,
        error_message="Corrupt PDF trailer",
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [failed_doc]
    mock_db.execute.return_value = mock_result

    res = await generate_document_summary(
        db=mock_db,
        user_id=user_id,
        session_id=session_id,
        knowledge_base_id=None,
        query="summarize this document",
    )
    assert "Processing failed for 'broken.pdf'" in res["final_response"]
    assert "Corrupt PDF trailer" in res["final_response"]
    assert "not covered" not in res["final_response"].lower()


@pytest.mark.asyncio
async def test_generate_summary_clarification_response():
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()
    doc1 = make_doc(uuid.uuid4(), user_id, "alpha.pdf", knowledge_base_id=kb_id)
    doc2 = make_doc(uuid.uuid4(), user_id, "beta.pdf", knowledge_base_id=kb_id)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [doc1, doc2]
    mock_db.execute.return_value = mock_result

    res = await generate_document_summary(
        db=mock_db,
        user_id=user_id,
        session_id=None,
        knowledge_base_id=kb_id,
        query="give me a summary",
    )
    assert "Which document would you like me to summarize?" in res["final_response"]
    assert "- alpha.pdf" in res["final_response"]
    assert "- beta.pdf" in res["final_response"]


# =====================================================================
# 5. Summarization Execution Tests (Single-Call vs Map-Reduce)
# =====================================================================

def make_chunk(chunk_id: uuid.UUID, doc_id: uuid.UUID, index: int, page: int, text: str) -> DocumentChunk:
    chunk = MagicMock(spec=DocumentChunk)
    chunk.id = chunk_id
    chunk.document_id = doc_id
    chunk.chunk_index = index
    chunk.page_number = page
    chunk.content = text
    return chunk


@pytest.mark.asyncio
async def test_generate_summary_single_call():
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    doc = make_doc(doc_id, user_id, "small_doc.pdf", session_id=session_id)

    chunks = [
        make_chunk(uuid.uuid4(), doc_id, 0, 1, "Page 1 content about project goals."),
        make_chunk(uuid.uuid4(), doc_id, 1, 2, "Page 2 content about timeline and deliverables."),
    ]

    mock_db = AsyncMock()
    # First call is resolve_target_document (scalars().all() -> [doc])
    # Second call is fetching chunks (scalars().all() -> chunks)
    doc_res = MagicMock()
    doc_res.scalars.return_value.all.return_value = [doc]

    chunk_res = MagicMock()
    chunk_res.scalars.return_value.all.return_value = chunks

    mock_db.execute.side_effect = [doc_res, chunk_res]

    with patch("ai.services.summarization_service.call_llm_with_backoff", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "### Overview\nThis document outlines project goals and timeline."
        res = await generate_document_summary(
            db=mock_db,
            user_id=user_id,
            session_id=session_id,
            knowledge_base_id=None,
            query="summarize this document",
        )

        assert mock_llm.call_count == 1
        assert "Overview" in res["final_response"]
        assert len(res["retrieved_chunks"]) == 2
        assert res["target_document"] == "small_doc.pdf"


@pytest.mark.asyncio
async def test_generate_summary_map_reduce_with_truncation():
    """Large documents that exceed single-call budget use map-reduce capped at 4 map + 1 reduce calls."""
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    doc = make_doc(doc_id, user_id, "huge_manual.pdf", session_id=session_id)

    # Create 6 large chunks of 25,000 characters each -> 150,000 characters total
    # Each chunk will form its own batch because 25,000 > BATCH_CHAR_TARGET (24,000)
    # Total batches = 6 > MAX_MAP_CALLS (4), so it should truncate to 4 map calls + 1 reduce call
    chunks = [
        make_chunk(uuid.uuid4(), doc_id, i, i + 1, f"Content page {i + 1} " + ("x" * 25000))
        for i in range(6)
    ]

    mock_db = AsyncMock()
    doc_res = MagicMock()
    doc_res.scalars.return_value.all.return_value = [doc]
    chunk_res = MagicMock()
    chunk_res.scalars.return_value.all.return_value = chunks
    mock_db.execute.side_effect = [doc_res, chunk_res]

    call_records = []

    async def mock_call_llm(prompt: str, max_retries: int = 2):
        call_records.append(prompt)
        if "Synthesize them into a unified" in prompt:
            return "Synthesized entire document summary."
        return f"Section summary {len(call_records)}"

    with patch("ai.services.summarization_service.call_llm_with_backoff", side_effect=mock_call_llm):
        res = await generate_document_summary(
            db=mock_db,
            user_id=user_id,
            session_id=session_id,
            knowledge_base_id=None,
            query="summarize this document",
        )

        # 4 map calls + 1 reduce call = 5 calls maximum
        assert len(call_records) == 5
        assert "Synthesized entire document summary." in res["final_response"]
        # Truncation note must be present
        assert "Due to document length constraints" in res["final_response"]
        assert "covers pages 1 to 4" in res["final_response"]


@pytest.mark.asyncio
async def test_generate_summary_rate_limit_handled():
    """If rate limit retries are exhausted, user receives friendly busy notification."""
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    doc = make_doc(doc_id, user_id, "doc.pdf", session_id=session_id)
    chunks = [make_chunk(uuid.uuid4(), doc_id, 0, 1, "Hello world")]

    mock_db = AsyncMock()
    doc_res = MagicMock()
    doc_res.scalars.return_value.all.return_value = [doc]
    chunk_res = MagicMock()
    chunk_res.scalars.return_value.all.return_value = chunks
    mock_db.execute.side_effect = [doc_res, chunk_res]

    with patch(
        "ai.services.summarization_service.call_llm_with_backoff",
        side_effect=RateLimitExceededError("Rate limit reached"),
    ):
        res = await generate_document_summary(
            db=mock_db,
            user_id=user_id,
            session_id=session_id,
            knowledge_base_id=None,
            query="summarize this document",
        )
        assert "AI service is currently busy due to rate limits" in res["final_response"]


# =====================================================================
# 6. Graph Routing & Workflow Tests
# =====================================================================

def test_route_intent_summarize():
    state = {"intent": "SUMMARIZE_DOCUMENT"}
    assert route_intent(state) == "summarize_document"


def test_route_after_answer_skips_reflection():
    state = {"intent": "SUMMARIZE_DOCUMENT"}
    assert route_after_answer(state) == "__end__"


@pytest.mark.asyncio
async def test_summarize_document_node_fail_closed_missing_user():
    """summarize_document_node fails closed when user_id is missing or invalid."""
    config = {"configurable": {"db": AsyncMock()}}

    # Missing user_id
    res1 = await summarize_document_node({"user_id": None}, config)
    assert "Could not verify user identity" in res1["final_response"]
    assert res1["retrieved_chunks"] == []

    # Invalid user_id
    res2 = await summarize_document_node({"user_id": "not-a-valid-uuid"}, config)
    assert "Invalid user identity format" in res2["final_response"]
    assert res2["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_graph_execution_summarize_path():
    """End-to-end graph routing for SUMMARIZE_DOCUMENT bypasses vector search and reflection."""
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    mock_db = AsyncMock()

    app = build_graph()

    summary_mock_output = {
        "final_response": "Full document summary here.",
        "retrieved_chunks": [{"filename": "doc.pdf", "page_number": 1}],
        "formatted_context": "[Summary]",
        "target_document": "doc.pdf",
    }

    initial_state = {
        "user_query": "summarize this document",
        "user_id": user_id,
        "session_id": session_id,
        "knowledge_base_id": None,
        "chat_history": [],
    }

    with patch(
        "ai.agents.graph.generate_document_summary",
        new_callable=AsyncMock,
        return_value=summary_mock_output,
    ):
        result = await app.ainvoke(
            initial_state,
            config={"configurable": {"db": mock_db}},
        )

        assert result["intent"] == "SUMMARIZE_DOCUMENT"
        assert result["final_response"] == "Full document summary here."
        # Reflection must NOT have run (verified remains None)
        assert result.get("verified") is None
        assert result.get("reflection_verdict") is None

