import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai.agents.supervisor import classify_intent, is_greeting, IntentClassification


@pytest.mark.parametrize(
    "query",
    [
        "hi",
        "Hi!",
        "hello",
        "hey there",
    ],
)
def test_is_greeting_positive(query: str):
    assert is_greeting(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "this document",
        "which model is used",
        "explain machine learning",
        "history of RAG",
        "this",
        "which",
        "architecture",
    ],
)
def test_is_greeting_negative(query: str):
    assert is_greeting(query) is False


@pytest.mark.asyncio
async def test_supervisor_intents():
    # 1. GREETING (handled by fast-path, no LLM required)
    state = {"user_query": "hello there"}
    res = await classify_intent(state)
    assert res["intent"] == "GREETING"

    # Mock the LLM call for the rest to avoid 429 Resource Exhausted errors
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("ai.agents.supervisor.llm_service.get_chat_model", return_value=mock_llm):
        # 2. DIRECT_ANSWER
        mock_structured.ainvoke.return_value = IntentClassification(intent="DIRECT_ANSWER", standalone_query="Write a python script to sort an array")
        state = {"user_query": "Write a python script to sort an array"}
        res = await classify_intent(state)
        assert res["intent"] == "DIRECT_ANSWER"

        # 3. WEB_SEARCH
        mock_structured.ainvoke.return_value = IntentClassification(intent="WEB_SEARCH", standalone_query="What is the AAPL stock price right now?")
        state = {"user_query": "What is the AAPL stock price right now?"}
        res = await classify_intent(state)
        assert res["intent"] == "WEB_SEARCH"

        # 4. RAG_QUERY
        mock_structured.ainvoke.return_value = IntentClassification(intent="RAG_QUERY", standalone_query="Summarize my uploaded notes")
        state = {"user_query": "Summarize my uploaded notes"}
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "hi",
        "Hi!",
        "hello",
        "hey there",
    ],
)
async def test_supervisor_fallback_greeting(query: str):
    # Pass chat_history to bypass fast-path and trigger the fallback path when LLM fails
    state = {
        "user_query": query,
        "chat_history": [{"role": "user", "content": "previous context"}],
    }
    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("LLM 429 Rate Limit")):
        res = await classify_intent(state)
        assert res["intent"] == "GREETING"
        assert res["standalone_query"] == query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "this document",
        "which model is used",
        "explain machine learning",
        "history of RAG",
        "this",
        "which",
        "architecture",
        "What does this document say about refunds?",
    ],
)
async def test_supervisor_fallback_rag_query(query: str):
    state = {"user_query": query}
    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("LLM 429 Rate Limit")):
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"
        assert res["standalone_query"] == query


@pytest.mark.asyncio
async def test_supervisor_fallback_with_image():
    state = {"user_query": "analyze this", "image_data": "data:image/png;base64,abc123"}
    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("LLM error")):
        res = await classify_intent(state)
        assert res["intent"] == "RAG_QUERY"
