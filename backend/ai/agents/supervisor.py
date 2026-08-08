import json
import logging
from ai.agents.state import AgentState
from ai.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

# Fast-path keyword matching for instant pleasantry detection
FAST_GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "sup",
    "who are you",
    "what can you do",
}


def is_simple_greeting(query: str) -> bool:
    cleaned = query.strip().lower()
    words = cleaned.split()
    if len(words) <= 3 and any(w in FAST_GREETING_KEYWORDS for w in words):
        return True
    return False


async def classify_intent(state: AgentState) -> str:
    """
    Classifies user intent into GREETING, RAG_QUERY, or WEB_SEARCH using fast keyword
    matching or LLM JSON evaluation.
    """
    user_query = state.get("user_query", "")
    if is_simple_greeting(user_query):
        return "GREETING"

    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {user_query}"},
    ]

    try:
        response_text = await llm_service.generate_response(
            messages=messages, temperature=0.0, max_tokens=150
        )

        # Sanitize response string for JSON parsing
        cleaned_response = (
            response_text.strip().removeprefix("```json").removesuffix("```").strip()
        )
        parsed = json.loads(cleaned_response)
        intent = parsed.get("intent", "RAG_QUERY").upper()

        if intent in ["GREETING", "RAG_QUERY", "WEB_SEARCH"]:
            return intent
        return "RAG_QUERY"
    except Exception as e:
        logger.warning(
            f"Intent classification parsing fallback triggered ({str(e)}). Defaulting to RAG_QUERY."
        )
        return "RAG_QUERY"