import base64
import logging
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Unified client service for LLM completions and streaming using Google Gemini
    via native LangChain integrations.
    """

    def __init__(self):
        # 1. Resolve and sanitize Gemini API Key from settings or environment
        raw_key = (
            getattr(settings, "GEMINI_API_KEY", None)
            or getattr(settings, "LLM_API_KEY", None)
            or getattr(settings, "OPENAI_API_KEY", None)
            or ""
        )
        self.api_key = str(raw_key).strip().strip("'\"")

        # 2. Resolve default model
        raw_model = getattr(settings, "LLM_MODEL", "gemini-1.5-flash")
        self.default_model = str(raw_model).strip().strip("'\"")

        # OpenAI-compatible base URL for image processing/vision endpoint
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        # Initialize AsyncOpenAI client for multimodal vision calls
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_chat_model(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
        streaming: bool = True,
    ) -> ChatGoogleGenerativeAI:
        """
        Returns a LangChain ChatGoogleGenerativeAI instance configured for Gemini.
        This enables LangGraph's astream_events to intercept on_chat_model_stream 
        for real SSE token streaming.
        """
        target_model = model or self.default_model
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=self.api_key,
            temperature=temperature,
            streaming=streaming,
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
        Uses ChatGoogleGenerativeAI to ensure compatibility with LangChain event systems.
        """
        target_model = model or self.default_model

        try:
            llm = self.get_chat_model(
                model=target_model,
                temperature=temperature,
                streaming=False,
            )

            # Convert dictionary messages to LangChain BaseMessage objects
            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            response = await llm.ainvoke(lc_messages)
            return str(response.content) if response and response.content else ""
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


# MUST BE AT THE BOTTOM: Global singleton instance exported for agents
llm_service = LLMService()