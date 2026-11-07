import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.agents.graph import rag_graph
from ai.agents.reflection import (
    UNVERIFIED_ANSWER_CAVEAT,
    evaluate_response,
    ReflectionOutput,
)


def _base_state(**overrides):
    state = {
        "session_id": str(uuid.uuid4()),
        "user_id": "test-user",
        "knowledge_base_id": str(uuid.uuid4()),
        "allow_web_search": False,
        "user_query": "What about Q3?",
        "standalone_query": "What were the company revenues in Q3 2024?",
        "image_data": None,
        "intent": "RAG_QUERY",
        "chat_history": [
            {"role": "user", "content": "What were the revenues in Q1 and Q2 2024?"},
            {"role": "assistant", "content": "Q1 revenue was $10M and Q2 revenue was $12M."},
        ],
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "filename": "financials.pdf",
                "content": "Q3 2024 total revenue reached $15M, an increase of 25% year-over-year.",
            }
        ],
        "formatted_context": "[Source: financials.pdf | Page 1]\nQ3 2024 total revenue reached $15M, an increase of 25% year-over-year.",
        "context_source": "document",
        "tool_outputs": [],
        "final_response": "In Q3 2024, total revenue reached $15M, up 25% year-over-year.",
        "error": None,
        "retry_count": 0,
        "previous_queries": [],
        "verified": None,
        "reflection_verdict": None,
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_reflection_receives_bounded_chat_history():
    """
    Verifies that the prompt sent to the LLM evaluator inside evaluate_response
    contains the recent chat history turns, enabling follow-up questions to be judged
    in conversational context.
    """
    state = _base_state()

    captured_prompt = None

    async def mock_ainvoke(messages):
        nonlocal captured_prompt
        # HumanMessage content contains eval_prompt
        captured_prompt = messages[1].content
        return ReflectionOutput(is_grounded=True, is_relevant=True, reason="Grounded in context and history")

    mock_structured = AsyncMock()
    mock_structured.ainvoke.side_effect = mock_ainvoke
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        result = await evaluate_response(state)

    assert result.get("error") is None
    assert result.get("verified") is True
    assert result.get("reflection_verdict") == "passed"
    assert captured_prompt is not None
    # Chat history must be present in the prompt
    assert "Recent Chat History:" in captured_prompt
    assert "What were the revenues in Q1 and Q2 2024?" in captured_prompt
    assert "$10M" in captured_prompt
    assert "User Query:\nWhat about Q3?" in captured_prompt


@pytest.mark.asyncio
async def test_reflection_chat_history_bounded():
    """
    Verifies that chat history is bounded to the most recent turns and truncated
    if excessively long.
    """
    long_history = [
        {"role": "user", "content": f"Historical question {i}: " + ("abc " * 50)}
        for i in range(20)
    ]
    state = _base_state(chat_history=long_history)

    captured_prompt = None

    async def mock_ainvoke(messages):
        nonlocal captured_prompt
        captured_prompt = messages[1].content
        return ReflectionOutput(is_grounded=True, is_relevant=True, reason="Passes")

    mock_structured = AsyncMock()
    mock_structured.ainvoke.side_effect = mock_ainvoke
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        await evaluate_response(state)

    assert captured_prompt is not None
    # Older turns beyond the recent 6 should not appear
    assert "Historical question 0:" not in captured_prompt
    assert "Historical question 1:" not in captured_prompt
    # Most recent turn should appear
    assert "Historical question 19:" in captured_prompt


@pytest.mark.asyncio
async def test_passing_answer_returned_unchanged_with_verified_true():
    """
    When reflection passes, verified must be True, reflection_verdict must be 'passed',
    and the final_response text must NOT be changed.
    """
    original_text = "In Q3 2024, total revenue reached $15M."
    state = _base_state(final_response=original_text)

    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=True,
        is_relevant=True,
        reason="Answer is fully supported.",
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        result = await evaluate_response(state)

    assert result.get("error") is None
    assert result.get("verified") is True
    assert result.get("reflection_verdict") == "passed"
    # final_response not modified in result
    assert "final_response" not in result or result["final_response"] == original_text
    assert UNVERIFIED_ANSWER_CAVEAT not in result.get("final_response", original_text)


@pytest.mark.asyncio
async def test_exhausted_retries_return_caveat_and_verified_false():
    """
    When the retry cap is reached and reflection still rejects the answer:
    - verified is False
    - reflection_verdict is 'failed'
    - final_response includes the user-visible caveat explaining it could not be fully verified.
    """
    original_draft = "The capital of Australia is Sydney according to common belief."
    state = _base_state(
        retry_count=2,  # Max retries reached
        final_response=original_draft,
    )

    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=False,
        is_relevant=False,
        reason="Sydney is not the capital; Canberra is.",
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        result = await evaluate_response(state)

    assert result.get("error") is not None
    assert result.get("verified") is False
    assert result.get("reflection_verdict") == "failed"
    returned_response = result.get("final_response", "")
    assert original_draft in returned_response
    assert UNVERIFIED_ANSWER_CAVEAT in returned_response
    assert "could not be fully verified" in returned_response


@pytest.mark.asyncio
async def test_end_to_end_graph_exhausted_retries_surfaces_caveat():
    """
    End-to-end graph test: when an answer continuously fails reflection across all retries,
    the final state from rag_graph.ainvoke must contain:
    - retry_count == 2
    - verified == False
    - reflection_verdict == 'failed'
    - final_response containing the unverified caveat
    """
    initial_state = _base_state(
        retry_count=0,
        final_response="",
    )

    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(
        is_grounded=False,
        is_relevant=False,
        reason="Draft never matches document evidence.",
    )
    mock_reflection_llm = MagicMock()
    mock_reflection_llm.with_structured_output.return_value = mock_structured

    with patch(
        "ai.agents.graph.classify_intent",
        new=AsyncMock(
            return_value={"intent": "RAG_QUERY", "standalone_query": "Q3 revenue"}
        ),
    ), patch(
        "ai.agents.graph.rewrite_query_for_retry",
        new=AsyncMock(side_effect=["Q3 revenue details", "Q3 financials report"]),
    ), patch(
        "ai.agents.graph.vector_search_tool",
        new=AsyncMock(
            return_value={"chunks": [{"content": "financial data"}], "context_text": "financial data"}
        ),
    ), patch(
        "ai.agents.answer.llm_service.generate_response",
        new=AsyncMock(return_value="Unverifiable draft about revenues."),
    ), patch(
        "ai.agents.reflection.llm_service.get_chat_model",
        return_value=mock_reflection_llm,
    ):
        final_state = await rag_graph.ainvoke(
            initial_state,
            config={"configurable": {"db": object()}, "recursion_limit": 25},
        )

    assert final_state["retry_count"] == 2
    assert final_state.get("verified") is False
    assert final_state.get("reflection_verdict") == "failed"
    assert "Unverifiable draft about revenues." in final_state["final_response"]
    assert UNVERIFIED_ANSWER_CAVEAT in final_state["final_response"]
