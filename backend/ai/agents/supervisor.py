import json
import logging
import re
from ai.agents.state import AgentState
from ai.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

# Fast-path keyword matching for instant detection
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

FAST_WEB_KEYWORDS = {
    "search the web",
    "web search",
    "latest news",
    "search online",
    "google",
    "browse the web",
    "latest updates",
}


def is_simple_greeting(query: str) -> bool:
    cleaned = query.strip().lower()
    words = cleaned.split()
    if len(words) <= 3 and any(w in FAST_GREETING_KEYWORDS for w in words):
        return True
    return False


def is_simple_web_search(query: str) -> bool:
    cleaned = query.strip().lower()
    return any(kw in cleaned for kw in FAST_WEB_KEYWORDS)


async def classify_intent(state: AgentState) -> str:
    """
    Classifies user intent into GREETING, RAG_QUERY, or WEB_SEARCH using fast keyword
    matching or LLM JSON evaluation.
    """
    user_query = state.get("user_query", "")

    # Fast-path checks
    if is_simple_greeting(user_query):
        return "GREETING"
    if is_simple_web_search(user_query):
        return "WEB_SEARCH"

    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {user_query}"},
    ]

    try:
        response_text = await llm_service.generate_response(
            messages=messages, temperature=0.0, max_tokens=150
        )

        # Extract JSON object using regex
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            intent = str(parsed.get("intent", "RAG_QUERY")).upper()
            if intent in ["GREETING", "RAG_QUERY", "WEB_SEARCH"]:
                return intent

        # Fallback string matching on response
        upper_resp = response_text.upper()
        if "WEB_SEARCH" in upper_resp:
            return "WEB_SEARCH"
        elif "GREETING" in upper_resp:
            return "GREETING"

        return "RAG_QUERY"
    except Exception as e:
        logger.warning(
            f"Intent classification parsing fallback triggered ({str(e)}). Defaulting to RAG_QUERY."
        )
        return "RAG_QUERY"