import base64
import logging
from typing import Any, Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config.settings import settings

logger = logging.getLogger(__name__)


def extract_text_from_content(content: Any) -> str:
    """
    Extracts clean text string from LangChain content outputs,
    handling plain strings, lists of text blocks, and metadata dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


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
        raw_model = getattr(settings, "LLM_MODEL", "gemini-3.5-flash")
        self.default_model = str(raw_model).strip().strip("'\"")

        # 3. Model Cache
        self._models: Dict[tuple, ChatGoogleGenerativeAI] = {}

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
        cache_key = (target_model, temperature, streaming)

        if cache_key not in self._models:
            self._models[cache_key] = ChatGoogleGenerativeAI(
                model=target_model,
                google_api_key=self.api_key,
                temperature=temperature,
                streaming=streaming,
            )
            
        return self._models[cache_key]

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
                
                # CRITICAL FIX: Do not str() lists to preserve multimodal image dicts
                raw_content = msg.get("content", "")
                if isinstance(raw_content, list):
                    content = raw_content
                else:
                    content = str(raw_content)

                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            response = await llm.ainvoke(lc_messages)
            return extract_text_from_content(response.content)
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
        Sends raw image bytes to Gemini via Langchain and returns a comprehensive textual breakdown.
        """
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
            return await self.generate_response(
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
                model=model
            )
        except Exception as e:
            logger.error(f"Gemini Vision API invocation failed: {str(e)}")
            raise RuntimeError(f"Vision service error: {str(e)}")


# MUST BE AT THE BOTTOM: Global singleton instance exported for agents
llm_service = LLMService()