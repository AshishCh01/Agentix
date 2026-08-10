import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai.agents.state import AgentState
from ai.agents.reflection import evaluate_response, ReflectionOutput
from ai.agents.answer import run_answer_agent

@pytest.fixture
def base_state() -> AgentState:
    return {
        "user_query": "What is the capital of France?",
        "formatted_context": "The capital of France is Paris.",
        "final_response": "The capital of France is Paris.",
        "intent": "RAG_QUERY",
        "chat_history": [],
        "tool_outputs": [],
        "retrieved_chunks": [],
        "session_id": "test",
        "user_id": "test",
        "image_data": None,
        "error": None,
        "retry_count": 0,
        "standalone_query": "What is the capital of France?"
    }

@pytest.mark.asyncio
async def test_reflection_pass(base_state):
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(is_grounded=True, is_relevant=True, reason="Looks good")
    mock_llm.with_structured_output.return_value = mock_structured
    
    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        res = await evaluate_response(base_state)
        assert res.get("error") is None

@pytest.mark.asyncio
async def test_reflection_fail(base_state):
    base_state["final_response"] = "The capital of France is London."
    
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(is_grounded=False, is_relevant=False, reason="London is not Paris")
    mock_llm.with_structured_output.return_value = mock_structured
    
    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        res = await evaluate_response(base_state)
        assert res.get("error") is not None
        assert "london" in res.get("error", "").lower() or "paris" in res.get("error", "").lower()

@pytest.mark.asyncio
async def test_reflection_retry(base_state):
    base_state["final_response"] = "The capital of France is London."
    
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ReflectionOutput(is_grounded=False, is_relevant=False, reason="London is not Paris")
    mock_llm.with_structured_output.return_value = mock_structured
    
    with patch("ai.agents.reflection.llm_service.get_chat_model", return_value=mock_llm):
        res_fail = await evaluate_response(base_state)
        
    base_state["error"] = res_fail.get("error")
    base_state["retry_count"] = 1
    
    with patch("ai.agents.answer.llm_service.generate_response", new_callable=AsyncMock) as mock_ans:
        mock_ans.return_value = "The capital of France is Paris."
        new_ans = await run_answer_agent(base_state)
        assert new_ans.get("final_response") != "The capital of France is London."
        assert "Paris" in new_ans.get("final_response", "")
