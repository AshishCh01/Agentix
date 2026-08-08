import json
import logging
import re
from ai.agents.state import AgentState
from ai.prompts.supervisor_prompt import SUPERVISOR_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

# Fast-path keyword sets
SHORT_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "howdy",
    "sup",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
}

CAPABILITY_PHRASES = {
    "who are you",
    "what can you do",
    "how can you help",
    "what are your features",
    "what is your purpose",
}

FAST_WEB_KEYWORDS = {
    "search the web",
    "web search",
    "latest news",
    "search online",
    "google",
    "browse the web",
    "latest updates",
    "who is",
    "what is",
    "where is",
    "latest",
    "python",
    "news",
    "current",
}


def is_simple_greeting(query: str) -> bool:
    cleaned = query.strip().lower()
    words = cleaned.split()

    # 1. Short greetings (3 words or fewer)
    if len(words) <= 3 and any(w.strip("!,.") in SHORT_GREETINGS for w in words):
        return True

    # 2. Identity or capability questions
    if any(phrase in cleaned for phrase in CAPABILITY_PHRASES):
        return True

    return False


def is_simple_web_search(query: str) -> bool:
    cleaned = query.strip().lower()
    return any(kw in cleaned for kw in FAST_WEB_KEYWORDS)


async def classify_intent(state: AgentState) -> str:
    """
    Classifies user intent into GREETING, RAG_QUERY, or WEB_SEARCH using fast keyword
    matching or LLM JSON evaluation with heuristic fallbacks.
    """
    user_query = state.get("user_query", "")

    # 1. Fast-path checks
    if is_simple_greeting(user_query):
        return "GREETING"
    if is_simple_web_search(user_query):
        return "WEB_SEARCH"

    # 2. LLM Classification
    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": f"User Query: {user_query}"},
    ]

    try:
        response_text = await llm_service.generate_response(
            messages=messages, temperature=0.0, max_tokens=150
        )

        # Clean markdown wrappers if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE).strip()

        # Extract JSON object using regex
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            intent = str(parsed.get("intent", "")).upper()
            if intent in ["GREETING", "RAG_QUERY", "WEB_SEARCH"]:
                return intent

        # Fallback string matching on LLM response text
        upper_resp = response_text.upper()
        if "WEB_SEARCH" in upper_resp:
            return "WEB_SEARCH"
        elif "GREETING" in upper_resp:
            return "GREETING"
        elif "RAG_QUERY" in upper_resp:
            return "RAG_QUERY"

    except Exception as e:
        logger.warning(
            f"Intent classification parsing fallback triggered ({str(e)}). Applying heuristic classification."
        )

    # 3. Intelligent Heuristic Fallback (Avoid defaulting everything to RAG_QUERY)
    q_lower = user_query.lower()
    if any(k in q_lower for k in ["search", "latest", "news", "who is", "what is", "python", "http", "www"]):
        return "WEB_SEARCH"
    elif any(k in q_lower for k in ["hi", "hello", "hey"]):
        return "GREETING"

    return "RAG_QUERY"