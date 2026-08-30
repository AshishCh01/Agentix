import asyncio

import pytest
from google import genai
from google.genai.errors import ClientError

from app.config.settings import settings


@pytest.mark.asyncio
async def test_gemini_connection():
    """
    Live connectivity smoke test for the configured Gemini API key/model.

    Previously this caught every exception, printed it, and returned
    normally -- so a missing key, an invalid key, or a totally dead
    connection all still reported PASSED. Now a broken key or a failed
    request actually fails the test. The one exception is a 429 quota
    error: that reflects the shared free-tier rate limit being exhausted,
    not a code or credential problem, so it's reported as skipped (visibly,
    not silently) rather than failing the whole suite over external quota
    exhaustion.
    """
    raw_key = settings.GEMINI_API_KEY
    assert raw_key, "GEMINI_API_KEY is empty in settings."

    client = genai.Client(api_key=raw_key)

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
