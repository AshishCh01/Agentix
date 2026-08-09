import logging
import uuid
from typing import Any, Dict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from ai.agents.answer import run_answer_agent
from ai.agents.greeting import run_greeting_agent
from ai.agents.reflection import evaluate_response
from ai.agents.state import AgentState
from ai.agents.supervisor import classify_intent
from ai.services.llm_service import llm_service
from ai.tools.vector_search import vector_search_tool
from ai.tools.web_search import web_search_tool
from app.config.settings import settings

logger = logging.getLogger(__name__)


# --- Helper Functions ---

async def contextualize_query(query: str, chat_history: list) -> str:
    """Rewrites short or ambiguous follow-up queries into standalone search queries using chat history."""
    clean_query = query.strip()

    if len(clean_query.split()) > 5 or not chat_history:
        return clean_query

    prompt = (
        "Given the following conversation history and a short follow-up user query, "
        "rephrase the follow-up query to be a complete, standalone search query. "
        "Do NOT answer the query—only output the rewritten standalone query string.\n\n"
        f"Chat History:\n{chat_history[-2:]}\n\n"
        f"Follow-up Query: {clean_query}\n"
        "Standalone Query:"
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        standalone_query = await llm_service.generate_response(
            messages=messages, temperature=0.0, max_tokens=50
        )
        return standalone_query.strip() or clean_query
    except Exception as e:
        logger.warning(f"Query contextualization failed: {e}")
        return clean_query


# --- 1. Node Definitions & Edge Functions ---

async def supervisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classifies user intent (GREETING, RAG_QUERY, or WEB_SEARCH)."""
    intent = await classify_intent(state)
    return {"intent": intent}


def route_intent(state: AgentState) -> str:
    """Conditional Edge router function for supervisor."""
    intent = state.get("intent", "RAG_QUERY")
    if intent == "GREETING":
        return "greeting"
    elif intent == "WEB_SEARCH":
        return "web_search"
    return "vector_search"


def route_after_vector_search(state: AgentState) -> str:
    """Routes to web_search if vector database returned zero chunks."""
    retrieved_chunks = state.get("retrieved_chunks", [])
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        logger.info("⚠️ Vector store returned 0 chunks. Rerouting to web_search fallback...")
        return "web_search"
    return "answer"


def route_after_answer(state: AgentState) -> str:
    """Conditional router: skips reflection if set in settings or for non-RAG queries."""
    if getattr(settings, "SKIP_REFLECTION", False):
        logger.info("⏩ Skipping reflection node (SKIP_REFLECTION=True).")
        return END

    intent = state.get("intent")
    if intent in ["WEB_SEARCH", "GREETING"]:
        logger.info(f"⏩ Skipping reflection node for intent: {intent}.")
        return END

    return "reflection"


def route_after_reflection(state: AgentState) -> str:
    """
    Self-correction loop: If reflection node flagged an ungrounded or irrelevant response,
    route back to answer_node to retry synthesis up to a max retry count.
    """
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if error and retry_count < max_retries:
        logger.info(
            f"🔄 [Reflection Self-Correction] Check failed ({error}). "
            f"Retrying answer generation (Attempt {retry_count + 1}/{max_retries})..."
        )
        return "answer"

    return END


async def greeting_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fast-path greeting generator node."""
    return await run_greeting_agent(state)


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
    chat_history = state.get("chat_history", [])
    search_query = await contextualize_query(raw_query, chat_history)

    if search_query != raw_query:
        logger.info(f"🔍 Rewrote query from '{raw_query}' to '{search_query}'")

    tool_result = await vector_search_tool(
        db=db,
        session_id=session_uuid,
        query=search_query,
        image_data=state.get("image_data"),
        top_k=4,
    )
    return {
        "retrieved_chunks": tool_result.get("chunks", []),
        "formatted_context": tool_result.get("context_text", ""),
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "vector_search", "result": tool_result}],
    }


async def web_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Web search engine retrieval node."""
    tool_result = await web_search_tool(query=state["user_query"], max_results=4)
    return {
        "formatted_context": tool_result.get("formatted_context", ""),
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
            "vector_search": "vector_search",
            "web_search": "web_search",
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

    # Self-Correction Edge from Reflection
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "answer": "answer",
            END: END,
        },
    )

    # Direct Node Edges
    workflow.add_edge("greeting", END)
    workflow.add_edge("web_search", "answer")

    return workflow.compile()


# Compiled Graph Instance
rag_graph = build_graph()