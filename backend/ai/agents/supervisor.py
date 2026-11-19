import logging
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ai.agents.state import AgentState
from ai.services.llm_service import llm_service
from ai.services.summarization_service import is_summarize_document_query
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """You are the Router Agent in an Agentic RAG system.
Your job is to categorize the user's query into an intent and rewrite conversational follow-ups into complete, self-contained standalone queries.

Intent Categories:
1. "GREETING": User says hi, hello, good morning, thanks, bye, or asks "who are you" / "what can you do" / personal questions about the assistant.
2. "RAG_QUERY": User asks about technical concepts, engineering topics, formulas, architectures, algorithms, course notes, definitions, summaries, key points, questions, or facts from uploaded PDFs, notes, or knowledge base documents. (DEFAULT for study/document queries, ANY technical engineering/ML/cloud/CV topic when a knowledge base is active, and ANY conversational follow-up discussing prior document answers).
3. "WEB_SEARCH": User explicitly asks about live internet data, latest real-time news, current weather, stock prices, events today, or explicitly asks to search the web/online.
4. "DIRECT_ANSWER": User asks for generic standalone coding assistance (e.g. "write a python snippet to sort a list"), basic math (e.g. "what is 25 * 14?"), casual chit-chat, logic puzzles, or creative writing that requires no external document or web lookup and does NOT ask about technical concepts in active study notes.
5. "SUMMARIZE_DOCUMENT": User asks for a complete summary, overview, tl;dr, or gist of an entire document, file, pdf, or upload (e.g. "summarize this document", "give me an overview of the pdf", "tl;dr of this file", "can you summarize the document I just uploaded?", "summarize handbook.pdf"). CRITICAL NOTE: Topic-specific requests (e.g. "summarize the section on refunds", "summarize what it says about pricing") seek information on a specific subject and MUST be classified as "RAG_QUERY", NOT "SUMMARIZE_DOCUMENT".

CRITICAL ROUTING RULE FOR KNOWLEDGE BASE / TECHNICAL QUERIES:
If a Knowledge Base is active (or user is in a document study session) and the question asks about a technical, engineering, scientific, machine learning, cloud computing, computer vision, or mathematical concept, you MUST classify the intent as "RAG_QUERY", NEVER "DIRECT_ANSWER". Even if the question could theoretically be answered from general AI training weights, technical questions must route to RAG_QUERY so the user's uploaded materials and notes are searched.

CRITICAL ROUTING RULE FOR FOLLOW-UPS:
If the user's query asks to explain, clarify, elaborate, compare, or continue discussing points from previous turns in the chat (e.g. "explain the second point in more detail", "what about Q3?", "tell me more about that"), you MUST classify the intent as "RAG_QUERY", NEVER "DIRECT_ANSWER".

Standalone Query Generation:
If the user's query is a follow-up or contains pronouns/ambiguous references (e.g. "it", "that", "the second one", "explain the second point in more detail", "what about Q3?", "how did he respond?"):
- Inspect the Chat History (both user questions and assistant answers).
- Resolve the referent into specific technical terms, nouns, numbered items, or topic headings from the conversation.
- Output a complete, self-contained query suitable for semantic vector retrieval against the knowledge base.
- If the query is already self-contained, output it exactly as is.

Few-Shot Examples:

Example 1 (Technical Computer Vision - RAG_QUERY):
User Query: How does stereo disparity map estimation determine depth from rectified epipolar geometry?
Intent: RAG_QUERY
Standalone Query: How does stereo disparity map estimation determine depth from rectified epipolar geometry?

Example 2 (Technical Cloud / Virtualization - RAG_QUERY):
User Query: What is the role of a hypervisor in hardware virtualization compared to OS-level container isolation?
Intent: RAG_QUERY
Standalone Query: What is the role of a hypervisor in hardware virtualization compared to OS-level container isolation?

Example 3 (Technical Sensor / Perception - RAG_QUERY):
User Query: How do perception pipelines align multi-sensor coordinate frames using extrinsic spatial calibration matrices?
Intent: RAG_QUERY
Standalone Query: How do perception pipelines align multi-sensor coordinate frames using extrinsic spatial calibration matrices?

Example 4 (Technical ML Optimization - RAG_QUERY):
User Query: How does the AdaGrad optimizer dynamically scale learning rates based on historical gradient norms?
Intent: RAG_QUERY
Standalone Query: How does the AdaGrad optimizer dynamically scale learning rates based on historical gradient norms?

Example 5 (Technical Cloud Networking - RAG_QUERY):
User Query: What are the fundamental throughput and latency trade-offs between virtual private clouds and software-defined WAN overlays?
Intent: RAG_QUERY
Standalone Query: What are the fundamental throughput and latency trade-offs between virtual private clouds and software-defined WAN overlays?

Example 6 (Technical Detection / Feature Extraction - RAG_QUERY):
User Query: How does non-maximum suppression (NMS) prune redundant bounding boxes generated by multi-scale anchor grids?
Intent: RAG_QUERY
Standalone Query: How does non-maximum suppression (NMS) prune redundant bounding boxes generated by multi-scale anchor grids?

Example 7 (Technical Optimization Math - RAG_QUERY):
User Query: What conditions guarantee that a constrained optimization objective satisfies Karush-Kuhn-Tucker (KKT) stationarity?
Intent: RAG_QUERY
Standalone Query: What conditions guarantee that a constrained optimization objective satisfies Karush-Kuhn-Tucker (KKT) stationarity?

Example 8 (Follow-up Technical RAG_QUERY):
Chat History:
User: What are the primary stages of data preprocessing in the pipeline?
Assistant: 1. Data ingestion and cleaning. 2. Feature normalization and scaling. 3. Categorical encoding.
User Query: explain the second point in more detail
Intent: RAG_QUERY
Standalone Query: Explain feature normalization and scaling in data preprocessing in more detail

Example 9 (GREETING):
User Query: Good morning! Can you help me study today?
Intent: GREETING
Standalone Query: Good morning! Can you help me study today?

Example 10 (DIRECT_ANSWER - general coding helper):
User Query: Write a quick Python function to reverse a linked list.
Intent: DIRECT_ANSWER
Standalone Query: Write a quick Python function to reverse a linked list.

Example 11 (DIRECT_ANSWER - simple math / casual conversation):
User Query: What is 45 multiplied by 18?
Intent: DIRECT_ANSWER
Standalone Query: What is 45 multiplied by 18?

Example 12 (WEB_SEARCH - live real-time request):
User Query: Search the web for today's top tech news headlines
Intent: WEB_SEARCH
Standalone Query: Today's top tech news headlines

Example 13 (SUMMARIZE_DOCUMENT - whole-document overview):
User Query: Can you summarize the entire document I just uploaded?
Intent: SUMMARIZE_DOCUMENT
Standalone Query: Can you summarize the entire document I just uploaded?
"""

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
    intent: Literal["GREETING", "RAG_QUERY", "WEB_SEARCH", "DIRECT_ANSWER", "SUMMARIZE_DOCUMENT"] = Field(
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

FOLLOW_UP_INDICATORS = re.compile(
    r"\b(it|its|that|this|these|those|the\s+(second|first|third|fourth|last|previous|other)\s*(one|point|item|step)?|"
    r"(second|first|third|fourth|last|previous)\s+(point|item|step|part|section)|"
    r"more\s+detail|explain\s+(further|more|in\s+detail)|"
    r"elaborate|what\s+about|why\s+is\s+that|how\s+about|tell\s+me\s+more|can\s+you\s+clarify|"
    r"how\s+does\s+that|what\s+does\s+that)\b",
    re.IGNORECASE,
)


def is_followup_query(query: str, chat_history: List[Dict[str, Any]]) -> bool:
    """
    Checks if a query appears to be a conversational follow-up referencing earlier turns.
    """
    if not chat_history:
        return False
    q_stripped = query.strip()
    if len(q_stripped.split()) <= 4:
        return True
    return bool(FOLLOW_UP_INDICATORS.search(q_stripped))


def build_fallback_standalone_query(query: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    Builds a safer standalone query fallback when LLM intent classification fails or times out.
    When chat history exists and the query is a follow-up, combines the last user question
    with the follow-up query to retain topic context for vector search.
    """
    if not chat_history or not is_followup_query(query, chat_history):
        return query

    # Find the most recent substantive user question in chat history
    last_user_question = None
    for msg in reversed(chat_history):
        if msg.get("role") == "user" and msg.get("content"):
            last_user_question = msg.get("content").strip()
            break

    if last_user_question:
        if query.lower() in last_user_question.lower():
            return last_user_question
        return f"{last_user_question} - {query}".strip()

    return query


WEB_CONFIRMATION_PATTERN = re.compile(
    r"^(?:"
    r"(?:yes|yeah|yep|yup|sure|sure\s+thing|ok|okay)?\s*,?\s*"
    r"(?:"
    r"please\s+do(?:\s+(?:that|so))?|"
    r"do\s+(?:that|it|so)|"
    r"go\s+ahead|"
    r"look\s+it\s+up(?:\s+online)?|"
    r"look\s+online|"
    r"(?:search|look\s+it\s+up)\s+(?:the\s+)?(?:web|online|internet)(?:\s+instead)?(?:\s+please)?|"
    r"please(?:\s+search(?:\s+(?:the\s+)?(?:web|online|internet))?)?|"
    r"search(?:\s+for\s+it)?"
    r")|"
    r"yes|yeah|yep|yup|sure|sure\s+thing|ok|okay|please"
    r")[\.\?!]*$",
    re.IGNORECASE,
)


def is_web_confirmation_phrase(query: str) -> bool:
    """
    Checks if a query is an affirmative confirmation phrase confirming web search.
    Examples: 'Yes, search the web', 'sure', 'please do', 'look it up online', 'yes please'.
    """
    if not query:
        return False
    return bool(WEB_CONFIRMATION_PATTERN.match(query.strip()))


def is_web_search_offer(text: str) -> bool:
    """
    Checks if an assistant message offered to search the web following an ungrounded or
    uncovered document query.
    """
    if not text:
        return False
    t = text.lower()
    return "would you like me to search the web" in t or (
        "couldn't find this in your uploaded documents" in t and "search the web" in t
    )


def get_last_assistant_message(chat_history: List[Dict[str, Any]]) -> Optional[str]:
    """
    Retrieves the content of the most recent assistant message from chat history.
    """
    if not chat_history:
        return None
    for msg in reversed(chat_history):
        role = msg.get("role") or msg.get("sender")
        if role == "assistant":
            return msg.get("content", "")
    return None


def get_previous_unanswered_question(chat_history: List[Dict[str, Any]]) -> Optional[str]:
    """
    Finds the user question that prompted the 'not covered' web-search offer.
    Searches backwards from the 'not covered' assistant message for the preceding user question.
    """
    if not chat_history:
        return None
    found_offer = False
    for msg in reversed(chat_history):
        role = msg.get("role") or msg.get("sender")
        content = (msg.get("content") or "").strip()
        if not found_offer:
            if role == "assistant" and is_web_search_offer(content):
                found_offer = True
        else:
            if role == "user" and content and not is_web_confirmation_phrase(content):
                return content
    return None


def get_last_non_confirmation_user_query(chat_history: List[Dict[str, Any]]) -> Optional[str]:
    """
    Finds the most recent user query that is not a confirmation phrase.
    """
    if not chat_history:
        return None
    for msg in reversed(chat_history):
        role = msg.get("role") or msg.get("sender")
        content = (msg.get("content") or "").strip()
        if role == "user" and content and not is_web_confirmation_phrase(content):
            return content
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

    # Fast-path 1: Deterministic web-search confirmation following a 'not covered' offer
    last_assistant_msg = get_last_assistant_message(chat_history)
    if (
        last_assistant_msg
        and is_web_search_offer(last_assistant_msg)
        and is_web_confirmation_phrase(user_query)
    ):
        original_query = (
            get_previous_unanswered_question(chat_history)
            or get_last_non_confirmation_user_query(chat_history)
        )
        logger.info(
            f"🌐 [Supervisor] Detected affirmative web search confirmation '{user_query}'. "
            f"Routing to WEB_SEARCH with original question: '{original_query}'"
        )
        return {
            "intent": "WEB_SEARCH",
            "standalone_query": original_query or user_query,
        }

    # Fast-path 2: Fast-path heuristic for simple greetings to bypass LLM latency
    if not chat_history and not image_data:
        q_lower = user_query.strip().lower()
        if q_lower in ["hi", "hello", "hey", "hi there", "hello there", "greetings"]:
            logger.info("⚡ Fast-path greeting detected. Bypassing LLM supervisor.")
            return {
                "intent": "GREETING",
                "standalone_query": user_query
            }

    # Fast-path 3: Deterministic whole-document summarization pre-check
    if is_summarize_document_query(user_query):
        logger.info(f"📄 [Supervisor] Deterministic whole-document summarization detected: '{user_query}'")
        return {
            "intent": "SUMMARIZE_DOCUMENT",
            "standalone_query": user_query,
        }

    # Build Multimodal Content for LLM Classification
    user_content: List[Dict[str, Any]] = []
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]])
        user_content.append({"type": "text", "text": f"Chat History:\n{history_str}\n\n"})

    if user_query.strip():
        user_content.append({"type": "text", "text": f"User Query: {user_query}"})
    else:
        user_content.append({"type": "text", "text": "User Query: [An image was attached without text]"})

    if image_data:
        image_url = image_data if image_data.startswith("data:") else f"data:image/png;base64,{image_data}"
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    system_prompt = SUPERVISOR_PROMPT
    if state.get("knowledge_base_id"):
        system_prompt += "\n\nACTIVE CONTEXT: A Knowledge Base is currently scoped and active for this session. Remember to default any technical concept, engineering, ML, cloud, CV, or study questions to RAG_QUERY."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    try:
        llm = llm_service.get_chat_model(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(IntentClassification)
        result = await structured_llm.ainvoke(messages)
        intent = result.intent
        standalone = result.standalone_query.strip() if result.standalone_query else user_query

        # Guard: follow-ups in knowledge-base chats or conversational follow-ups referencing
        # prior document answers must route to RAG_QUERY, never DIRECT_ANSWER.
        if (
            intent == "DIRECT_ANSWER"
            and chat_history
            and (state.get("knowledge_base_id") or is_followup_query(user_query, chat_history))
        ):
            logger.info("🔄 Reclassifying conversational follow-up from DIRECT_ANSWER to RAG_QUERY.")
            intent = "RAG_QUERY"
            if standalone == user_query:
                standalone = build_fallback_standalone_query(user_query, chat_history)

        # Guard: if LLM classified as WEB_SEARCH but left standalone_query as a confirmation phrase,
        # or if user confirmed web search after a 'not covered' offer, resolve it to original query.
        if intent == "WEB_SEARCH" and (
            is_web_confirmation_phrase(standalone)
            or (last_assistant_msg and is_web_search_offer(last_assistant_msg) and is_web_confirmation_phrase(user_query))
        ):
            resolved_q = (
                get_previous_unanswered_question(chat_history)
                or get_last_non_confirmation_user_query(chat_history)
            )
            if resolved_q:
                logger.info(f"🌐 Resolved generic WEB_SEARCH standalone query to: '{resolved_q}'")
                standalone = resolved_q

        return {
            "intent": intent,
            "standalone_query": standalone
        }
    except Exception as e:
        logger.warning(
            f"Intent classification structured output failed ({str(e)}). Applying fallback."
        )
        if image_data:
            fallback_intent = "RAG_QUERY"
            fallback_query = user_query
        elif (
            last_assistant_msg
            and is_web_search_offer(last_assistant_msg)
            and is_web_confirmation_phrase(user_query)
        ):
            fallback_intent = "WEB_SEARCH"
            fallback_query = (
                get_previous_unanswered_question(chat_history)
                or get_last_non_confirmation_user_query(chat_history)
                or user_query
            )
        elif is_greeting(user_query):
            fallback_intent = "GREETING"
            fallback_query = user_query
        elif is_summarize_document_query(user_query):
            fallback_intent = "SUMMARIZE_DOCUMENT"
            fallback_query = user_query
        else:
            fallback_intent = "RAG_QUERY"
            fallback_query = build_fallback_standalone_query(user_query, chat_history)

        return {
            "intent": fallback_intent,
            "standalone_query": fallback_query
        }


async def run_supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph entry node wrapper for supervisor routing.
    """
    return await classify_intent(state)