import uuid

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from ai.agents.graph import rag_graph
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


@pytest.mark.asyncio
async def test_reflection_loop_stops_at_max_retry_cap():
    """
    Graph-level regression test: a response that NEVER passes reflection must
    not loop supervisor -> vector_search -> answer -> reflection forever.
    route_after_reflection's hardcoded max_retries=2 has to kick in and route
    to END once retry_count reaches it, giving exactly 3 total passes (the
    original attempt plus 2 retries). Exercised through rag_graph.ainvoke --
    only the outbound LLM/tool calls are mocked, so the real routing/retry
    bookkeeping in graph.py and answer.py/reflection.py runs unmodified. A
    recursion_limit is set as a safety net: if the cap logic regressed into
    an infinite loop, this fails loudly with GraphRecursionError instead of
    hanging.
    """
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=False,
        is_relevant=False,
        reason="Never grounded -- forces the retry loop to exhaust its cap",
    )
    mock_reflection_llm = MagicMock()
    mock_reflection_llm.with_structured_output.return_value = mock_structured

    mock_generate_response = AsyncMock(
        return_value="A response that will never satisfy reflection."
    )

    initial_state = _base_state(intent=None)

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "refund policy"}
        ),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        new=AsyncMock(
            return_value={"chunks": [{"content": "refund info"}], "context_text": "refund info"}
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response", new=mock_generate_response
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model", return_value=mock_reflection_llm
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    # The cap must actually have been hit -- not exceeded, not silently reset.
    assert final_state["retry_count"] == 2
    assert final_state.get("error") is not None

    # Exactly 3 full passes: the initial attempt plus the 2 retries the cap
    # allows. If the loop terminated early this undercounts; if the cap were
    # broken it would keep climbing (or hit the recursion_limit instead).
    assert mock_generate_response.call_count == 3
    assert mock_structured.ainvoke.call_count == 3

    # And it must have gone back through the supervisor/vector_search path
    # each retry, not just re-run answer/reflection in place.
    reflection_entries = [
        t
        for t in final_state.get("tool_outputs", [])
        if isinstance(t, dict) and t.get("node") == "reflection"
    ]
    assert len(reflection_entries) == 3
    assert all(entry["is_grounded"] is False for entry in reflection_entries)
