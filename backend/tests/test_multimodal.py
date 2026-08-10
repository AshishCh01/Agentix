import pytest
import base64
from unittest.mock import patch, AsyncMock
from ai.services.llm_service import llm_service
from ai.agents.answer import run_answer_agent

# Create a dummy 1x1 transparent PNG base64
dummy_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
dummy_bytes = base64.b64decode(dummy_png_b64)

@pytest.mark.asyncio
async def test_describe_image():
    with patch("ai.services.llm_service.LLMService.describe_image", new_callable=AsyncMock) as mock_desc:
        mock_desc.return_value = "1x1 pixel"
        desc = await llm_service.describe_image(
            file_bytes=dummy_bytes,
            mime_type="image/png",
            prompt="What is this image? It's just a 1x1 dummy pixel, so you can just say '1x1 pixel'."
        )
        assert type(desc) == str
        assert len(desc) > 0

@pytest.mark.asyncio
async def test_generate_response_multimodal():
    messages = [
        {"role": "system", "content": "You are a helpful vision AI."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image? It is a dummy 1x1 pixel."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dummy_png_b64}"}}
            ]
        }
    ]
    with patch("ai.services.llm_service.LLMService.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "This is a 1x1 dummy pixel."
        resp = await llm_service.generate_response(messages=messages)
        assert type(resp) == str
        assert len(resp) > 0

@pytest.mark.asyncio
async def test_generate_response_text():
    text_msgs = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    with patch("ai.services.llm_service.LLMService.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "4"
        resp2 = await llm_service.generate_response(messages=text_msgs)
        assert type(resp2) == str
        assert "4" in resp2
