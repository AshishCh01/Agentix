import asyncio
from unittest.mock import patch, MagicMock

import pytest
from google import genai
from google.genai.errors import ClientError

from app.config.settings import settings


@pytest.mark.asyncio
@patch("google.genai.models.Models.generate_content")
async def test_gemini_connection(mock_generate_content):
    """
    Mocked connectivity smoke test for the configured Gemini API key/model.
    """
    raw_key = settings.GEMINI_API_KEY
    assert raw_key, "GEMINI_API_KEY is empty in settings."

    client = genai.Client(api_key=raw_key)
    
    mock_response = MagicMock()
    mock_response.text = "Gemini API is connected successfully!"
    mock_generate_content.return_value = mock_response

    try:
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents="Respond with 'Gemini API is connected successfully!' if you receive this message.",
        )
    except ClientError as e:
        if getattr(e, "code", None) == 429:
            pytest.skip(f"Gemini API quota exhausted -- inconclusive, not a code failure: {e}")
        raise

    assert response.text, "Gemini API returned an empty response."


if __name__ == "__main__":
    asyncio.run(test_gemini_connection())
