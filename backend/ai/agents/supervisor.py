import logging
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ai.agents.state import AgentState
from ai.prompts.supervisor_prompt import SUPERVISOR_PROMPT
from ai.services.llm_service import llm_service
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

GREETING_PATTERN = re.compile(r"\b(hi|hello|hey)\b", re.IGNORECASE)


def is_greeting(query: str) -> bool:
    """
    Check if the query contains a greeting as a whole word.
    Avoids matching substrings in words like 'this', 'which', 'machine', 'history', 'architecture'.
    """
    if not query:
        return False
    return bool(GREETING_PATTERN.search(query))


class IntentClassification(BaseModel):
    intent: Literal["GREETING", "RAG_QUERY", "WEB_SEARCH", "DIRECT_ANSWER"] = Field(
        description="The classified intent of the user query."
    )
    standalone_query: str = Field(
        description="The query rewritten to be self-contained using chat history. If already self-contained, this should be identical to the original user query."
    )


class RetryQueryRewrite(BaseModel):
    rewritten_query: str = Field(
        description="The newly refined search query targeting missing or corrected information from reflection feedback."
    )


RETRY_REWRITE_SYSTEM_PROMPT = """You are a Search Query Refiner in an Agentic RAG system.
A previous search for the user's question retrieved document context that led to an incomplete, inaccurate, or ungrounded draft answer.
Based on the original question and the reflection feedback explaining what was missing or incorrect, generate a NEW, targeted retrieval query.

Guidelines:
1. Focus on keywords, concepts, or missing details identified in the reflection feedback.
2. The query must be different from previous queries.
3. Keep the query concise, keyword-rich, and natural for semantic/hybrid document search.
4. Do not include quotes, conversational preamble, or explanations.
"""

STOP_WORDS_SET = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "not", "this", "that", "these", "those", "it", "its",
    "with", "as", "by", "from", "be", "has", "have", "had", "does", "do", "did",
    "response", "context", "grounded", "relevant", "evaluator", "failed", "check",
    "reflection", "error", "user", "query", "assistant", "provided"
}


def build_deterministic_retry_query(
    original_query: str,
    reflection_feedback: str,
    previous_queries: List[str],
) -> Optional[str]:
    """
    Deterministic fallback to generate a new retrieval query when LLM rewrite
    fails or produces a duplicate query. Extracts novel keywords from reflection
    feedback and combines them with the original query.
    """
    clean_feedback = re.sub(
        r"^(Reflection Check Failed|Evaluator Error):\s*",
        "",
        reflection_feedback,
        flags=re.IGNORECASE,
    ).strip()

    tokens = re.findall(r"[A-Za-z0-9_-]+", clean_feedback)
    orig_tokens_lower = set(re.findall(r"[A-Za-z0-9_-]+", original_query.lower()))

    novel_keywords = [
        t for t in tokens
        if t.lower() not in STOP_WORDS_SET and t.lower() not in orig_tokens_lower
    ]

    candidates: List[str] = []
    if novel_keywords:
        candidates.append(f"{original_query} {' '.join(novel_keywords[:6])}".strip())
        candidates.append(" ".join(novel_keywords[:8]).strip())

    if clean_feedback:
        candidates.append(f"{original_query} {clean_feedback[:100]}".strip())
        candidates.append(clean_feedback[:120].strip())

    prev_lower = {q.strip().lower() for q in previous_queries if q}
    for cand in candidates:
        cand_clean = cand.strip()
        if (
            cand_clean
            and cand_clean.lower() not in prev_lower
            and cand_clean.lower() != original_query.strip().lower()
        ):
            return cand_clean

    return None


async def rewrite_query_for_retry(
    state: AgentState,
    original_query: str,
    reflection_feedback: str,
    previous_queries: List[str],
) -> Optional[str]:
    """
    Rewrites the retrieval query on a reflection retry using reflection feedback.
    Uses LLM structured output with a deterministic fallback if the rewrite fails
    or produces a query identical to one already tried.
    """
    prev_lower = {q.strip().lower() for q in previous_queries if q}

    user_prompt = f"""Original Question:
{original_query}

Reflection Feedback (what was missing or wrong):
{reflection_feedback}

Previously Tried Queries:
{', '.join(f'"{q}"' for q in previous_queries)}

Generate a new, distinct retrieval query targeting the missing information."""

    messages = [
        SystemMessage(content=RETRY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        llm = llm_service.get_chat_model(temperature=0.2, streaming=False)
        structured_llm = llm.with_structured_output(RetryQueryRewrite)
        result = await structured_llm.ainvoke(messages)
        candidate = result.rewritten_query.strip().strip('"\'')
        if (
            candidate
            and candidate.lower() not in prev_lower
            and candidate.lower() != original_query.strip().lower()
        ):
            logger.info(f"🔄 [Query Rewrite] LLM generated new retrieval query: '{candidate}'")
            return candidate
        logger.warning(
            f"⚠️ [Query Rewrite] LLM returned duplicate or empty query ('{candidate}'). "
            "Falling back to deterministic rewrite."
        )
    except Exception as e:
        logger.warning(
            f"⚠️ [Query Rewrite] LLM query rewrite failed ({e}). Falling back to deterministic rewrite."
        )

    fallback_query = build_deterministic_retry_query(
        original_query=original_query,
        reflection_feedback=reflection_feedback,
        previous_queries=previous_queries,
    )
    if fallback_query:
        logger.info(f"🔄 [Query Rewrite] Deterministic fallback query generated: '{fallback_query}'")
        return fallback_query

    return None

async def classify_intent(state: AgentState) -> Dict[str, Any]:
    """
    Classifies user intent into GREETING, RAG_QUERY, WEB_SEARCH, or DIRECT_ANSWER
    using native structured output from the LLM, and generates a standalone query.
    Returns a dict with 'intent' and 'standalone_query'.
    """
    user_query = state.get("user_query", "")
    image_data = state.get("image_data")
    chat_history = state.get("chat_history", [])

    # Fast-path heuristic for simple greetings to bypass LLM latency
    if not chat_history and not image_data:
        q_lower = user_query.strip().lower()
        if q_lower in ["hi", "hello", "hey", "hi there", "hello there", "greetings"]:
            logger.info("⚡ Fast-path greeting detected. Bypassing LLM supervisor.")
            return {
                "intent": "GREETING",
                "standalone_query": user_query
            }

    # Build Multimodal Content for LLM Classification
    user_content: List[Dict[str, Any]] = []
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]])
        user_content.append({"type": "text", "text": f"Chat History:\n{history_str}\n\n"})

    if user_query.strip():
        user_content.append({"type": "text", "text": f"User Query: {user_query}"})
    else:
        user_content.append({"type": "text", "text": "User Query: [An image was attached without text]"})

    if image_data:
        image_url = image_data if image_data.startswith("data:") else f"data:image/png;base64,{image_data}"
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        llm = llm_service.get_chat_model(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(IntentClassification)
        result = await structured_llm.ainvoke(messages)
        return {
            "intent": result.intent,
            "standalone_query": result.standalone_query
        }
    except Exception as e:
        logger.warning(
            f"Intent classification structured output failed ({str(e)}). Applying fallback."
        )
        if image_data:
            fallback_intent = "RAG_QUERY"
        elif is_greeting(user_query):
            fallback_intent = "GREETING"
        else:
            fallback_intent = "RAG_QUERY"
            
        return {
            "intent": fallback_intent,
            "standalone_query": user_query
        }


async def run_supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph entry node wrapper for supervisor routing.
    """
    return await classify_intent(state)