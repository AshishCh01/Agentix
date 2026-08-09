import base64
import logging
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Unified client service for LLM completions using Google Gemini's
    OpenAI-compatible API endpoint.
    """

    def __init__(self):
        # 1. Resolve Gemini API Key from settings or environment
        self.api_key = (
            getattr(settings, "GEMINI_API_KEY", None)
            or getattr(settings, "LLM_API_KEY", None)
            or getattr(settings, "OPENAI_API_KEY", None)
            or ""
        )

        # 2. Google Gemini OpenAI-compatible base URL
        self.base_url = getattr(
            settings,
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # 3. Default model
        self.default_model = getattr(settings, "LLM_MODEL", "gemini-3.5-flash")

        # Initialize AsyncOpenAI client pointing to Google's Gemini endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        """
        Sends formatted conversation messages to Gemini and returns the completion string.
        """
        target_model = model or self.default_model

        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Gemini LLM API invocation failed: {str(e)}")
            raise RuntimeError(f"LLM service error: {str(e)}")

    async def describe_image(
        self,
        file_bytes: bytes,
        mime_type: str = "image/png",
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Sends raw image bytes to Gemini Vision endpoint and returns a comprehensive textual breakdown.
        """
        target_model = model or self.default_model
        base64_image = base64.b64encode(file_bytes).decode("utf-8")

        default_prompt = (
            "Provide a detailed and comprehensive textual breakdown of this image. "
            "Extract all readable text word-for-word, describe all diagrams, charts, "
            "flowcharts, architecture, visual elements, and tables, and summarize the key "
            "information accurately for vector database indexing."
        )
        image_prompt = prompt or default_prompt

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Gemini Vision API invocation failed: {str(e)}")
            raise RuntimeError(f"Vision service error: {str(e)}")


# Global singleton instance
llm_service = LLMService()