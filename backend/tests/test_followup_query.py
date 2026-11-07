import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai.agents.supervisor import (
    classify_intent,
    is_followup_query,
    build_fallback_standalone_query,
    IntentClassification,
)


def test_is_followup_query():
    history = [
        {"role": "user", "content": "What are the stages of data preprocessing?"},
        {"role": "assistant", "content": "1. Ingestion 2. Scaling 3. Encoding"},
    ]

    # Positive cases
    assert is_followup_query("explain the second point in more detail", history) is True
    assert is_followup_query("tell me more about that", history) is True
    assert is_followup_query("what about it?", history) is True
    assert is_followup_query("why is that?", history) is True
    assert is_followup_query("can you clarify?", history) is True
    assert is_followup_query("more detail", history) is True  # short query <= 4 words

    # Negative cases
    assert is_followup_query("explain the second point", []) is False  # No history
    assert (
        is_followup_query(
            "Calculate the determinant of a 3x3 matrix using Gaussian elimination steps",
            history,
        )
        is False
    )


def test_build_fallback_standalone_query():
    history = [
        {"role": "user", "content": "What are the primary stages of data preprocessing in the pipeline?"},
        {"role": "assistant", "content": "1. Ingestion. 2. Feature normalization and scaling. 3. Categorical encoding."},
    ]

    # Follow-up with history combines previous user question + query
    fallback = build_fallback_standalone_query("explain the second point in more detail", history)
    assert fallback == "What are the primary stages of data preprocessing in the pipeline? - explain the second point in more detail"

    # Query already contained in previous user question
    assert build_fallback_standalone_query("data preprocessing", history) == "What are the primary stages of data preprocessing in the pipeline?"

    # No history returns original query
    assert build_fallback_standalone_query("explain the second point", []) == "explain the second point"

    # History with only assistant message returns original query
    assistant_only_history = [{"role": "assistant", "content": "Hello!"}]
    assert build_fallback_standalone_query("explain the second point", assistant_only_history) == "explain the second point"


@pytest.mark.asyncio
async def test_classify_intent_followup_guard_reroutes_direct_answer():
    """
    If LLM mistakenly returns DIRECT_ANSWER for a conversational follow-up,
    the supervisor guard must reroute it to RAG_QUERY.
    """
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    history = [
        {"role": "user", "content": "What is the return policy for hardware purchases?"},
        {"role": "assistant", "content": "Hardware can be returned within 30 days."},
    ]

    with patch("ai.agents.supervisor.llm_service.get_chat_model", return_value=mock_llm):
        # Case 1: LLM returns DIRECT_ANSWER on follow-up query with history
        mock_structured.ainvoke.return_value = IntentClassification(
            intent="DIRECT_ANSWER",
            standalone_query="explain the second point in more detail",
        )
        state = {
            "user_query": "explain the second point in more detail",
            "chat_history": history,
            "knowledge_base_id": "kb-123",
        }
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"
        # Standalone query should be enriched by fallback if LLM left it as raw query
        assert "What is the return policy for hardware purchases?" in res["standalone_query"]

        # Case 2: Standalone coding question with no history and no KB stays DIRECT_ANSWER
        mock_structured.ainvoke.return_value = IntentClassification(
            intent="DIRECT_ANSWER",
            standalone_query="Write a python function to reverse a linked list",
        )
        res_direct = await classify_intent({"user_query": "Write a python function to reverse a linked list"})
        assert res_direct["intent"] == "DIRECT_ANSWER"
        assert res_direct["standalone_query"] == "Write a python function to reverse a linked list"


@pytest.mark.asyncio
async def test_classify_intent_followup_fallback_on_exception():
    """
    When LLM classification fails (e.g. timeout or API error), follow-up queries
    must fall back safely to RAG_QUERY with an enriched standalone query.
    """
    history = [
        {"role": "user", "content": "Explain transformer self-attention mechanism."},
        {"role": "assistant", "content": "Self-attention computes dot products between query and key vectors."},
    ]

    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("API timeout")):
        # Follow-up query with history
        state = {
            "user_query": "explain the second point in more detail",
            "chat_history": history,
        }
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"
        assert res["standalone_query"] == "Explain transformer self-attention mechanism. - explain the second point in more detail"

        # Standalone query with NO history
        state_no_history = {
            "user_query": "What is the capital of France?",
            "chat_history": [],
        }
        res_no_history = await classify_intent(state_no_history)
        assert res_no_history["intent"] == "RAG_QUERY"
        assert res_no_history["standalone_query"] == "What is the capital of France?"


@pytest.mark.asyncio
async def test_classify_intent_standalone_query_resolution():
    """
    Verify that when LLM returns an enriched standalone query, supervisor uses it.
    """
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    history = [
        {"role": "user", "content": "What are the primary stages of data preprocessing in the pipeline?"},
        {"role": "assistant", "content": "1. Data ingestion. 2. Feature normalization and scaling. 3. Categorical encoding."},
    ]

    with patch("ai.agents.supervisor.llm_service.get_chat_model", return_value=mock_llm):
        mock_structured.ainvoke.return_value = IntentClassification(
            intent="RAG_QUERY",
            standalone_query="Explain feature normalization and scaling in data preprocessing in more detail",
        )
        state = {
            "user_query": "explain the second point in more detail",
            "chat_history": history,
        }
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"
        assert res["standalone_query"] == "Explain feature normalization and scaling in data preprocessing in more detail"
