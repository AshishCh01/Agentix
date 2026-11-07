import logging
import uuid
from typing import Any, Dict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from ai.agents.answer import run_answer_agent
from ai.agents.greeting import run_greeting_agent
from ai.agents.direct_answer import run_direct_answer_agent
from ai.agents.reflection import UNVERIFIED_ANSWER_CAVEAT, evaluate_response
from ai.agents.state import AgentState
from ai.agents.supervisor import (
    classify_intent,
    rewrite_query_for_retry,
    is_web_confirmation_phrase,
    get_previous_unanswered_question,
    get_last_non_confirmation_user_query,
)
from ai.services.llm_service import llm_service
from ai.tools.vector_search import vector_search_tool
from ai.tools.web_search import web_search_tool
from app.config.settings import settings

logger = logging.getLogger(__name__)


# --- Helper Functions ---
# (contextualize_query removed as supervisor handles it now)



# --- 1. Node Definitions & Edge Functions ---

async def supervisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classifies user intent (GREETING, RAG_QUERY, WEB_SEARCH, DIRECT_ANSWER) or rewrites query on retry."""
    is_retry = bool(state.get("error"))
    previous_queries = list(state.get("previous_queries") or [])

    if is_retry:
        original_query = state.get("standalone_query") or state.get("user_query", "")
        if original_query and original_query not in previous_queries:
            previous_queries.append(original_query)

        feedback = state.get("error", "")
        new_query = await rewrite_query_for_retry(
            state=state,
            original_query=original_query,
            reflection_feedback=feedback,
            previous_queries=previous_queries,
        )

        if not new_query or new_query in previous_queries:
            logger.warning(
                f"⚠️ [Supervisor Node] Could not produce a new query distinct from previous queries "
                f"({previous_queries}). Stopping retry loop."
            )
            final_resp = state.get("final_response", "")
            if UNVERIFIED_ANSWER_CAVEAT not in final_resp:
                final_resp = f"{final_resp.rstrip()}{UNVERIFIED_ANSWER_CAVEAT}"
            return {
                "intent": "STOP_RETRY",
                "final_response": final_resp,
                "verified": False,
                "reflection_verdict": "failed",
                "previous_queries": previous_queries,
            }

        previous_queries.append(new_query)
        logger.info(
            f"🔄 [Supervisor Node] Rewrote retrieval query for retry: '{new_query}' "
            f"(previous: {previous_queries[:-1]})"
        )
        return {
            "intent": "RAG_QUERY",
            "standalone_query": new_query,
            "previous_queries": previous_queries,
        }

    # Initial pass: classify intent
    result = await classify_intent(state)
    initial_query = result.get("standalone_query") or state.get("user_query", "")
    if initial_query and initial_query not in previous_queries:
        previous_queries.append(initial_query)

    result["previous_queries"] = previous_queries
    return result


def route_intent(state: AgentState) -> str:
    """Conditional Edge router function for supervisor."""
    intent = state.get("intent", "RAG_QUERY")
    if intent == "GREETING":
        return "greeting"
    elif intent == "WEB_SEARCH":
        return "web_search"
    elif intent == "DIRECT_ANSWER":
        return "direct_answer"
    elif intent == "STOP_RETRY":
        return END
    return "vector_search"


def route_after_vector_search(state: AgentState) -> str:
    """
    Routes after vector retrieval based on chunk relevance and knowledge base scoping.
    """
    if state.get("retrieval_error"):
        logger.info("⚠️ Retrieval error occurred. Routing to answer to handle error gracefully.")
        return "answer"
    
    retrieved_chunks = state.get("retrieved_chunks", [])
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        # If this is a knowledge-base chat, 0 chunks passing threshold means documents do
        # not cover the query. Web search is opt-in (Decision 2 & 3).
        if state.get("knowledge_base_id"):
            if state.get("allow_web_search"):
                logger.info("⚠️ 0 chunks passed threshold in KB chat, but allow_web_search is enabled. Routing to web_search...")
                return "web_search"
            logger.info("⚠️ 0 chunks passed threshold in KB chat. Routing to answer for 'not covered' response (web search is opt-in).")
            return "answer"
        
        # In non-KB chat (e.g. ad-hoc session chat with no KB attached), preserve existing fallback
        logger.info("⚠️ Vector store returned 0 chunks in non-KB chat. Rerouting to web_search fallback...")
        return "web_search"

    return "answer"


def route_after_answer(state: AgentState) -> str:
    """
    Conditional router: skips reflection if set in settings, for non-RAG queries,
    or when 0 chunks were retrieved (the 'not covered' path).

    `intent` here reflects the supervisor's original classification, not
    whether web_search happened to run — web_search_node no longer overwrites
    it. So a deliberate WEB_SEARCH query still skips reflection, but a
    RAG_QUERY that fell back to web search (empty vector retrieval) keeps its
    RAG_QUERY intent and correctly falls through to reflection below.
    """
    if getattr(settings, "SKIP_REFLECTION", False):
        logger.info("⏩ Skipping reflection node (SKIP_REFLECTION=True).")
        return END

    intent = state.get("intent")
    if intent in ["WEB_SEARCH", "GREETING", "DIRECT_ANSWER"]:
        logger.info(f"⏩ Skipping reflection node for intent: {intent}.")
        return END

    # Guard for Issue #10: The "not covered" path (0 chunks after thresholding)
    # has nothing to verify and must NOT enter reflection or the retry loop.
    final_resp = state.get("final_response", "")
    if "I couldn't find this in your uploaded documents" in final_resp:
        logger.info("⏩ Skipping reflection node for 'not covered' response.")
        return END

    if state.get("context_source") != "web" and not state.get("retrieved_chunks") and not state.get("image_data"):
        logger.info("⏩ Skipping reflection node: no document chunks passed relevance threshold ('not covered' response).")
        return END

    if state.get("retrieval_error"):
        logger.info("⏩ Skipping reflection node: retrieval error occurred.")
        return END

    return "reflection"


def route_after_reflection(state: AgentState) -> str:
    """
    Self-correction loop: Routes back to SUPERVISOR instead of answer
    so the system can re-attempt retrieval or web search with fresh context.
    """
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if error and retry_count < max_retries:
        logger.info(
            f"🔄 [Reflection Self-Correction] Check failed ({error}). "
            f"Retrying from supervisor (Attempt {retry_count + 1}/{max_retries})..."
        )
        return "supervisor"

    return END


async def greeting_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fast-path greeting generator node."""
    return await run_greeting_agent(state)


async def direct_answer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Direct answer agent node for general knowledge/coding queries."""
    return await run_direct_answer_agent(state)


async def vector_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Vector database retrieval node with query contextualization."""
    configurable = config.get("configurable", {}) if config else {}
    db = configurable.get("db")

    if db is None:
        raise ValueError("AsyncSession 'db' was not provided in RunnableConfig['configurable']")

    session_uuid = (
        state["session_id"]
        if isinstance(state["session_id"], uuid.UUID)
        else uuid.UUID(state["session_id"])
    )

    raw_query = state.get("user_query", "")
    search_query = state.get("standalone_query") or raw_query

    if search_query != raw_query:
        logger.info(f"🔍 Using standalone query from supervisor: '{search_query}'")

    kb_id = state.get("knowledge_base_id")
    tool_result = await vector_search_tool(
        db=db,
        session_id=session_uuid,
        query=search_query,
        image_data=state.get("image_data"),
        top_k=4,
        knowledge_base_id=uuid.UUID(kb_id) if kb_id else None,
    )
    
    retrieval_error = tool_result.get("retrieval_error")
    
    return {
        "retrieval_error": bool(retrieval_error),
        "retrieval_error_message": retrieval_error if isinstance(retrieval_error, str) else None,
        "retrieved_chunks": tool_result.get("chunks", []),
        "formatted_context": tool_result.get("context_text", ""),
        # Reset every RAG pass so a stale "web" value from an earlier
        # reflection-retry iteration can't leak into this one.
        "context_source": "document",
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "vector_search", "result": tool_result}],
    }


async def web_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Web search engine retrieval node."""
    search_query = state.get("standalone_query") or state.get("user_query", "")
    chat_history = state.get("chat_history", [])

    if is_web_confirmation_phrase(search_query):
        fallback_query = (
            get_previous_unanswered_question(chat_history)
            or get_last_non_confirmation_user_query(chat_history)
        )
        if fallback_query:
            logger.info(
                f"🌐 [Web Search Node] Search query '{search_query}' was a confirmation phrase; "
                f"falling back to '{fallback_query}'."
            )
            search_query = fallback_query
        else:
            logger.warning(
                f"⚠️ [Web Search Node] Search query '{search_query}' was a confirmation phrase "
                "but no prior user question was found in chat history."
            )
            return {
                "context_source": "web",
                "formatted_context": "",
                "retrieved_chunks": [],
                "final_response": "What would you like me to search the web for? Please provide a specific question or topic.",
                "tool_outputs": state.get("tool_outputs", []) + [{
                    "tool": "web_search",
                    "result": {"error": "Missing search query: generic confirmation without prior question"},
                }],
            }

    tool_result = await web_search_tool(query=search_query, max_results=4)
    return {
        # NOTE: intent is deliberately NOT overwritten here. Leaving the
        # original classification intact means a genuine WEB_SEARCH intent
        # (set by the supervisor) still skips reflection in route_after_answer,
        # while a RAG_QUERY that fell back to web search here keeps its
        # original intent and still goes through reflection/groundedness
        # checking downstream — context_source is what tells answer_node
        # this content came from the web rather than retrieved documents.
        "context_source": "web",
        "formatted_context": tool_result.get("formatted_context", ""),
        # This mapping is crucial: it passes the web URLs to your frontend citations
        "retrieved_chunks": tool_result.get("sources", []),
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "web_search", "result": tool_result}],
    }


async def answer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Answer synthesis agent node."""
    current_retry = state.get("retry_count", 0)
    if state.get("error"):
        current_retry += 1

    res = await run_answer_agent(state)
    res["retry_count"] = current_retry
    return res


async def reflection_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Reflection & groundedness evaluator node."""
    return await evaluate_response(state)


# --- 2. Build StateGraph Workflow ---

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("reflection", reflection_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Routing Edges
    workflow.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "greeting": "greeting",
            "direct_answer": "direct_answer",
            "vector_search": "vector_search",
            "web_search": "web_search",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "vector_search",
        route_after_vector_search,
        {
            "web_search": "web_search",
            "answer": "answer",
        },
    )

    workflow.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "reflection": "reflection",
            END: END,
        },
    )

    # Self-Correction Edge routing to Supervisor instead of Answer
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    # Direct Node Edges
    workflow.add_edge("greeting", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("web_search", "answer")

    return workflow.compile()


# Compiled Graph Instance
rag_graph = build_graph()