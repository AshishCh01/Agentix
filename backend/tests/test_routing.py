import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai.agents.supervisor import classify_intent, IntentClassification

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


