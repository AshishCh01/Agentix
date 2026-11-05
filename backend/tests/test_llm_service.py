import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage
from ai.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_generate_response_passes_custom_max_tokens():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Custom token response"))

    with patch("ai.services.llm_service.ChatGoogleGenerativeAI", return_value=mock_model) as mock_cls:
        service = LLMService()
        messages = [{"role": "user", "content": "Hello world"}]
        response = await service.generate_response(
            messages=messages,
            max_tokens=2048,
        )

        assert response == "Custom token response"
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("max_output_tokens") == 2048


@pytest.mark.asyncio
async def test_generate_response_passes_default_max_tokens():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Default token response"))

    with patch("ai.services.llm_service.ChatGoogleGenerativeAI", return_value=mock_model) as mock_cls:
        service = LLMService()
        messages = [{"role": "user", "content": "Hello world"}]
        response = await service.generate_response(
            messages=messages,
        )

        assert response == "Default token response"
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("max_output_tokens") == 1024


def test_get_chat_model_max_output_tokens():
    mock_model = MagicMock()
    with patch("ai.services.llm_service.ChatGoogleGenerativeAI", return_value=mock_model) as mock_cls:
        service = LLMService()
        service.get_chat_model(max_tokens=500)
        _, kwargs = mock_cls.call_args
        assert kwargs.get("max_output_tokens") == 500


def test_get_chat_model_default_no_max_tokens():
    mock_model = MagicMock()
    with patch("ai.services.llm_service.ChatGoogleGenerativeAI", return_value=mock_model) as mock_cls:
        service = LLMService()
        service.get_chat_model()
        _, kwargs = mock_cls.call_args
        assert "max_output_tokens" not in kwargs
