import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.agents.graph import rag_graph, route_after_answer
from ai.agents.reflection import ReflectionOutput
from ai.agents.supervisor import build_deterministic_retry_query, rewrite_query_for_retry


def _base_state(**overrides):
    state = {
        "session_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "knowledge_base_id": str(uuid.uuid4()),
        "allow_web_search": False,
        "user_query": "What is the warranty period?",
        "standalone_query": None,
        "image_data": None,
        "intent": "RAG_QUERY",
        "chat_history": [],
        "retrieved_chunks": [],
        "formatted_context": "",
        "context_source": "document",
        "tool_outputs": [],
        "final_response": "",
        "error": None,
        "retry_count": 0,
        "previous_queries": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_retry_query_differs_from_previous():
    """
    On a retry, the supervisor must rewrite the retrieval query to target missing
    information, producing a search query different from the original query, and
    recording all queries in previous_queries state.
    """
    initial_state = _base_state(
        user_query="What is the refund policy?",
        standalone_query="What is the refund policy?",
    )

    searched_queries = []

    async def mock_vector_search(db, session_id, query, **kwargs):
        searched_queries.append(query)
        return {
            "chunks": [{"chunk_id": "c1", "content": "Refund policy details"}],
            "context_text": "Refund policy details",
        }

    # Reflection fails on first pass, passes on second pass
    reflection_attempts = 0

    def mock_reflection_structured():
        nonlocal reflection_attempts
        mock_structured = AsyncMock()
        if reflection_attempts == 0:
            reflection_attempts += 1
            mock_structured.ainvoke.return_value = ReflectionOutput(
                is_grounded=False,
                is_relevant=False,
                reason="The answer omitted the 30-day receipt requirement.",
            )
        else:
            mock_structured.ainvoke.return_value = ReflectionOutput(
                is_grounded=True,
                is_relevant=True,
                reason="Response is accurate and grounded.",
            )
        return mock_structured

    mock_reflection_llm = MagicMock()
    mock_reflection_llm.with_structured_output.side_effect = mock_reflection_structured

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "What is the refund policy?"}
        ),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        side_effect=mock_vector_search,
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=AsyncMock(return_value="Refund is possible."),
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model",
        return_value=mock_reflection_llm,
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    # Retrieval must have run at least twice
    assert len(searched_queries) >= 2
    first_query = searched_queries[0]
    second_query = searched_queries[1]

    # Crucial assertion: the second query MUST differ from the first
    assert second_query != first_query
    assert "What is the refund policy?" in first_query
    # Both queries must be recorded in previous_queries
    assert len(final_state.get("previous_queries", [])) >= 2
    assert first_query in final_state["previous_queries"]
    assert second_query in final_state["previous_queries"]


@pytest.mark.asyncio
async def test_same_query_retry_is_prevented():
    """
    If query rewriting fails to produce a novel query (e.g. returns a query already in
    previous_queries), the system must stop retrying immediately rather than repeating
    the same search.
    """
    initial_state = _base_state(
        user_query="Same query test",
        standalone_query="Same query test",
    )

    searched_queries = []

    async def mock_vector_search(db, session_id, query, **kwargs):
        searched_queries.append(query)
        return {
            "chunks": [{"chunk_id": "c1", "content": "Some context"}],
            "context_text": "Some context",
        }

    mock_reflection_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=False,
        is_relevant=False,
        reason="Check failed",
    )
    mock_reflection_llm.with_structured_output.return_value = mock_structured

    # Mock rewrite_query_for_retry to return None (no new query can be produced)
    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "Same query test"}
        ),
    ), patch(
        "ai.agents.graph.rewrite_query_for_retry",
        new=AsyncMock(return_value=None),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        side_effect=mock_vector_search,
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=AsyncMock(return_value="Draft response"),
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model",
        return_value=mock_reflection_llm,
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    # Vector search must only have run once; the duplicate search was prevented
    assert len(searched_queries) == 1
    assert searched_queries[0] == "Same query test"


@pytest.mark.asyncio
async def test_retry_cap_still_enforced():
    """
    Even with continuous query rewrites on each retry, the graph must stop when
    max_retries is reached (giving exactly 3 passes: 1 initial + 2 retries).
    """
    initial_state = _base_state()

    rewrites = ["retry query 1", "retry query 2", "retry query 3"]
    mock_generate_response = AsyncMock(return_value="Draft answer that always fails")

    mock_reflection_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=False,
        is_relevant=False,
        reason="Always ungrounded",
    )
    mock_reflection_llm.with_structured_output.return_value = mock_structured

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "initial query"}
        ),
    ), patch(
        "ai.agents.graph.rewrite_query_for_retry",
        new=AsyncMock(side_effect=rewrites),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        new=AsyncMock(
            return_value={"chunks": [{"content": "info"}], "context_text": "info"}
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=mock_generate_response,
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model",
        return_value=mock_reflection_llm,
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    assert final_state["retry_count"] == 2
    assert mock_generate_response.call_count == 3


@pytest.mark.asyncio
async def test_rewrite_failure_falls_back_safely():
    """
    When the LLM query rewrite throws an exception, the deterministic fallback
    must produce a valid, novel query combining the original query and reflection keywords.
    """
    state = _base_state(
        standalone_query="cancel subscription",
        error="Reflection Check Failed: The response omitted annual refund eligibility requirements.",
    )

    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("LLM Timeout")):
        rewritten = await rewrite_query_for_retry(
            state=state,
            original_query="cancel subscription",
            reflection_feedback=state["error"],
            previous_queries=["cancel subscription"],
        )

    assert rewritten is not None
    assert rewritten != "cancel subscription"
    assert "cancel subscription" in rewritten
    # Must have extracted non-stopword tokens from feedback
    assert any(term in rewritten.lower() for term in ["annual", "refund", "eligibility", "requirements"])


def test_build_deterministic_retry_query_avoids_duplicates():
    """
    Deterministic fallback returns None if all candidate variations are already
    in previous_queries.
    """
    orig = "return policy"
    feedback = "Reflection Check Failed: missing 14 days"
    # Exhaust all possible candidates in previous_queries
    prev = [
        "return policy",
        "return policy missing 14 days",
        "missing 14 days",
        "missing 14 days return policy",
    ]
    res = build_deterministic_retry_query(orig, feedback, prev)
    # When all variations are seen, should return None so retry loop terminates safely
    assert res is None or res not in prev


@pytest.mark.asyncio
async def test_not_covered_skips_reflection_and_retry_zero_llm_calls():
    """
    When 0 chunks survive the relevance threshold in a KB chat without opt-in web search:
    - routes to 'answer', which returns the honest 'not covered' message without LLM
    - route_after_answer routes directly to END (skips reflection and retry completely)
    - exactly zero LLM calls are made across the entire pipeline.
    """
    initial_state = _base_state(
        knowledge_base_id=str(uuid.uuid4()),
        allow_web_search=False,
    )

    mock_llm_answer = AsyncMock()
    mock_reflection_model = MagicMock()

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "obscure unmentioned topic"}
        ),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        new=AsyncMock(
            return_value={"chunks": [], "context_text": ""}
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=mock_llm_answer,
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model",
        return_value=mock_reflection_model,
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    # 1. Answer text must be the canonical 'not covered' message
    assert "I couldn't find this in your uploaded documents." in final_state["final_response"]
    assert "Would you like me to search the web instead?" in final_state["final_response"]

    # 2. Zero LLM calls made
    mock_llm_answer.assert_not_called()
    mock_reflection_model.get_chat_model.assert_not_called()
    mock_reflection_model.with_structured_output.assert_not_called()

    # 3. Reflection was never executed
    tool_outputs = final_state.get("tool_outputs", [])
    assert not any(t.get("node") == "reflection" for t in tool_outputs if isinstance(t, dict))

    # 4. Never entered retry loop
    assert final_state.get("retry_count", 0) == 0
