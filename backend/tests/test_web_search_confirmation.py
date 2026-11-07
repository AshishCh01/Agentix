import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai.agents.supervisor import (
    classify_intent,
    is_web_confirmation_phrase,
    is_web_search_offer,
    get_previous_unanswered_question,
    get_last_non_confirmation_user_query,
    IntentClassification,
)
from ai.agents.graph import web_search_node, rag_graph, route_after_vector_search


def test_is_web_confirmation_phrase():
    positives = [
        "Yes, search the web",
        "yes, search the web.",
        "sure",
        "please do",
        "look it up online",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "yes please",
        "sure thing",
        "go ahead",
        "search the web",
        "search online",
        "yes, search online please",
        "please search the web",
        "look it up",
        "search the internet instead",
        "please do that",
        "yes please do",
        "sure, go ahead",
        "yes, do that",
    ]
    for phrase in positives:
        assert is_web_confirmation_phrase(phrase) is True, f"Failed for {phrase}"

    negatives = [
        "What is Apple's revenue?",
        "Yes, search for quantum computing papers on arxiv",
        "tell me more about that",
        "explain the second point",
        "",
        "no, thanks",
    ]
    for phrase in negatives:
        assert is_web_confirmation_phrase(phrase) is False, f"Failed negative for {phrase}"


def test_is_web_search_offer():
    offer_msg = (
        "I couldn't find this in your uploaded documents. "
        "Would you like me to search the web instead?"
    )
    assert is_web_search_offer(offer_msg) is True
    assert is_web_search_offer("Would you like me to search the web?") is True
    assert is_web_search_offer("Here are the document chunks retrieved.") is False
    assert is_web_search_offer("") is False


def test_history_extraction_helpers():
    history = [
        {"role": "user", "content": "What is the return policy for electronics?"},
        {
            "role": "assistant",
            "content": "I couldn't find this in your uploaded documents. Would you like me to search the web instead?",
        },
    ]

    assert (
        get_previous_unanswered_question(history)
        == "What is the return policy for electronics?"
    )
    assert (
        get_last_non_confirmation_user_query(history)
        == "What is the return policy for electronics?"
    )

    # Empty history
    assert get_previous_unanswered_question([]) is None
    assert get_last_non_confirmation_user_query([]) is None

    # History where user already typed a confirmation
    history_with_confirm = history + [{"role": "user", "content": "yes, search the web"}]
    assert (
        get_last_non_confirmation_user_query(history_with_confirm)
        == "What is the return policy for electronics?"
    )


@pytest.mark.asyncio
async def test_supervisor_web_confirmation_fast_path():
    """
    Typed confirmation after 'not covered' offer immediately classifies as
    WEB_SEARCH with original query without needing LLM.
    """
    history = [
        {"role": "user", "content": "What is the capital of Atlantis?"},
        {
            "role": "assistant",
            "content": "I couldn't find this in your uploaded documents. Would you like me to search the web instead?",
        },
    ]

    for phrase in ["Yes, search the web", "sure", "please do", "look it up online"]:
        state = {
            "user_query": phrase,
            "chat_history": history,
            "knowledge_base_id": "kb-test",
        }
        res = await classify_intent(state)
        assert res["intent"] == "WEB_SEARCH"
        assert res["standalone_query"] == "What is the capital of Atlantis?"


@pytest.mark.asyncio
async def test_supervisor_web_confirmation_fallback_on_exception():
    """
    Even if LLM throws an error/timeout, affirmative confirmation safely falls back to WEB_SEARCH
    with the original user question.
    """
    history = [
        {"role": "user", "content": "Where is the nearest branch?"},
        {
            "role": "assistant",
            "content": "I couldn't find this in your uploaded documents. Would you like me to search the web instead?",
        },
    ]

    with patch("ai.agents.supervisor.llm_service.get_chat_model", side_effect=Exception("API timeout")):
        state = {
            "user_query": "sure thing",
            "chat_history": history,
        }
        res = await classify_intent(state)
        assert res["intent"] == "WEB_SEARCH"
        assert res["standalone_query"] == "Where is the nearest branch?"


@pytest.mark.asyncio
async def test_web_search_node_safety_check_resolves_query():
    """
    If web_search_node receives a confirmation phrase, it falls back to the prior user question
    instead of searching the confirmation string.
    """
    history = [
        {"role": "user", "content": "Explain superconducting qubits"},
        {
            "role": "assistant",
            "content": "I couldn't find this in your uploaded documents. Would you like me to search the web instead?",
        },
    ]

    mock_search_tool = AsyncMock(return_value={
        "formatted_context": "Web results on superconducting qubits",
        "sources": [{"title": "Qubits", "url": "https://example.com"}],
    })

    with patch("ai.agents.graph.web_search_tool", new=mock_search_tool):
        state = {
            "user_query": "yes, search the web",
            "standalone_query": "yes, search the web",  # literal phrase
            "chat_history": history,
        }
        result = await web_search_node(state, config={})
        # Check tool was called with the original question, not 'yes, search the web'
        mock_search_tool.assert_called_once_with(query="Explain superconducting qubits", max_results=4)
        assert result["context_source"] == "web"
        assert result["formatted_context"] == "Web results on superconducting qubits"


@pytest.mark.asyncio
async def test_web_search_node_no_prior_question_returns_clarification():
    """
    If web_search_node receives a confirmation phrase and NO prior user question exists in history,
    it returns a clear clarification message and never searches junk.
    """
    mock_search_tool = AsyncMock()

    with patch("ai.agents.graph.web_search_tool", new=mock_search_tool):
        state = {
            "user_query": "yes, search the web",
            "standalone_query": "yes, search the web",
            "chat_history": [],  # No history
        }
        result = await web_search_node(state, config={})
        # Tool MUST NOT be called
        mock_search_tool.assert_not_called()
        assert "What would you like me to search the web for?" in result["final_response"]


def test_allow_web_search_flag_path_preserved():
    """
    Regression check: 0 chunks in KB chat with allow_web_search=True still routes to web_search.
    """
    state_opt_in = {
        "retrieved_chunks": [],
        "knowledge_base_id": "kb-123",
        "allow_web_search": True,
    }
    assert route_after_vector_search(state_opt_in) == "web_search"

    state_no_opt_in = {
        "retrieved_chunks": [],
        "knowledge_base_id": "kb-123",
        "allow_web_search": False,
    }
    assert route_after_vector_search(state_no_opt_in) == "answer"

