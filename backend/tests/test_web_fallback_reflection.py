import uuid

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from ai.agents.graph import (
    rag_graph,
    route_after_answer,
    route_after_vector_search,
    web_search_node,
    vector_search_node,
)
from ai.agents.reflection import ReflectionOutput


def _base_state(**overrides):
    state = {
        "session_id": str(uuid.uuid4()),
        "user_id": "test-user",
        "user_query": "What is the refund policy?",
        "standalone_query": None,
        "image_data": None,
        "intent": "RAG_QUERY",
        "chat_history": [],
        "retrieved_chunks": [],
        "formatted_context": "",
        "context_source": None,
        "tool_outputs": [],
        "final_response": "",
        "error": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


# --- Unit tests on the pure routing functions ---


def test_route_after_vector_search_falls_back_on_empty_chunks():
    assert route_after_vector_search(_base_state(retrieved_chunks=[])) == "web_search"


def test_route_after_vector_search_goes_to_answer_when_chunks_found():
    assert route_after_vector_search(_base_state(retrieved_chunks=[{"content": "x"}])) == "answer"


def test_route_after_answer_reflects_on_rag_query_even_after_web_fallback():
    # This is the core bug: a RAG_QUERY that fell back to web search must
    # still keep its original intent and go through reflection.
    state = _base_state(intent="RAG_QUERY", context_source="web")
    assert route_after_answer(state) == "reflection"


def test_route_after_answer_skips_reflection_for_direct_web_search():
    # A query the supervisor classified as WEB_SEARCH directly should still
    # skip reflection, preserving intentional behavior.
    state = _base_state(intent="WEB_SEARCH", context_source="web")
    from langgraph.graph import END
    assert route_after_answer(state) == END


def test_route_after_answer_skips_reflection_for_greeting_and_direct_answer():
    from langgraph.graph import END
    assert route_after_answer(_base_state(intent="GREETING")) == END
    assert route_after_answer(_base_state(intent="DIRECT_ANSWER")) == END


# --- Node-level tests ---


@pytest.mark.asyncio
async def test_web_search_node_does_not_overwrite_intent():
    mock_result = {
        "formatted_context": "Source: Example\nContent: refunds within 30 days.",
        "sources": [{"filename": "Example", "url": "https://example.com", "chunk_index": 0}],
        "results": [],
    }
    with patch("ai.agents.graph.web_search_tool", new=AsyncMock(return_value=mock_result)):
        state = _base_state(intent="RAG_QUERY")
        result = await web_search_node(state, config={})

    # intent must NOT appear in the returned partial state at all — the node
    # should never touch it.
    assert "intent" not in result
    assert result["context_source"] == "web"


@pytest.mark.asyncio
async def test_vector_search_node_sets_document_context_source():
    mock_result = {"chunks": [{"content": "refund info"}], "context_text": "refund info"}
    with patch("ai.agents.graph.vector_search_tool", new=AsyncMock(return_value=mock_result)):
        state = _base_state(intent="RAG_QUERY")
        result = await vector_search_node(state, config={"configurable": {"db": object()}})

    assert result["context_source"] == "document"


# --- Full graph regression test ---


@pytest.mark.asyncio
async def test_rag_query_web_fallback_runs_through_reflection_end_to_end():
    """
    Regression test for the audit finding: RAG_QUERY -> empty vector search
    -> web_search fallback must still be groundedness-checked by reflection,
    not silently exempted the way a direct WEB_SEARCH intent is.
    """
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=True, is_relevant=True, reason="Matches web content"
    )
    mock_reflection_llm = MagicMock()
    mock_reflection_llm.with_structured_output.return_value = mock_structured

    initial_state = _base_state(intent=None)

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(return_value={"intent": "RAG_QUERY", "standalone_query": "refund policy"}),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        new=AsyncMock(return_value={"chunks": [], "context_text": ""}),
    ), patch(
        "ai.agents.graph.web_search_tool",
        new=AsyncMock(
            return_value={
                "formatted_context": "Source: Refund Policy\nContent: Refunds within 30 days.",
                "sources": [
                    {
                        "filename": "Refund Policy",
                        "url": "https://example.com",
                        "chunk_index": 0,
                        "similarity_score": 1.0,
                        "page_number": None,
                    }
                ],
                "results": [],
            }
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=AsyncMock(return_value="You can get a refund within 30 days."),
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model", return_value=mock_reflection_llm
    ):
        final_state = await rag_graph.ainvoke(
            initial_state, config={"configurable": {"db": object()}}
        )

    # The original classification survives the web fallback.
    assert final_state["intent"] == "RAG_QUERY"
    assert final_state["context_source"] == "web"

    # Reflection actually ran (not skipped) and recorded its verdict.
    tool_outputs = final_state.get("tool_outputs", [])
    reflection_entries = [
        t for t in tool_outputs if isinstance(t, dict) and t.get("node") == "reflection"
    ]
    assert len(reflection_entries) == 1
    assert reflection_entries[0]["is_grounded"] is True
    assert final_state.get("error") is None


@pytest.mark.asyncio
async def test_direct_web_search_intent_still_skips_reflection_end_to_end():
    """
    A query the supervisor classifies as WEB_SEARCH directly (not a fallback)
    must keep its existing behavior of skipping reflection.
    """
    initial_state = _base_state(intent=None)

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(return_value={"intent": "WEB_SEARCH", "standalone_query": "AAPL stock price"}),
    ), patch(
        "ai.agents.graph.web_search_tool",
        new=AsyncMock(
            return_value={
                "formatted_context": "Source: Stock Ticker\nContent: AAPL is at $200.",
                "sources": [
                    {
                        "filename": "Stock Ticker",
                        "url": "https://example.com",
                        "chunk_index": 0,
                        "similarity_score": 1.0,
                        "page_number": None,
                    }
                ],
                "results": [],
            }
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=AsyncMock(return_value="AAPL is currently at $200."),
    ):
        final_state = await rag_graph.ainvoke(
            initial_state, config={"configurable": {"db": object()}}
        )

    assert final_state["intent"] == "WEB_SEARCH"
    assert final_state["context_source"] == "web"

    tool_outputs = final_state.get("tool_outputs", [])
    reflection_entries = [
        t for t in tool_outputs if isinstance(t, dict) and t.get("node") == "reflection"
    ]
    assert len(reflection_entries) == 0
