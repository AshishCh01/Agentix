import logging
import uuid
from typing import Any, Dict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from ai.agents.answer import run_answer_agent
from ai.agents.greeting import run_greeting_agent
from ai.agents.direct_answer import run_direct_answer_agent
from ai.agents.reflection import evaluate_response
from ai.agents.state import AgentState
from ai.agents.supervisor import classify_intent
from ai.services.llm_service import llm_service
from ai.tools.vector_search import vector_search_tool
from ai.tools.web_search import web_search_tool
from app.config.settings import settings

logger = logging.getLogger(__name__)


# --- Helper Functions ---
# (contextualize_query removed as supervisor handles it now)



# --- 1. Node Definitions & Edge Functions ---

async def supervisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classifies user intent (GREETING, RAG_QUERY, WEB_SEARCH, DIRECT_ANSWER)."""
    result = await classify_intent(state)
    # classify_intent returns {"intent": "...", "standalone_query": "..."}
    # Return it directly so both keys are merged into graph state
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
    return "vector_search"


def route_after_vector_search(state: AgentState) -> str:
    """Routes to web_search if vector database returned zero chunks."""
    retrieved_chunks = state.get("retrieved_chunks", [])
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        logger.info("⚠️ Vector store returned 0 chunks. Rerouting to web_search fallback...")
        return "web_search"
    return "answer"


def route_after_answer(state: AgentState) -> str:
    """
    Conditional router: skips reflection if set in settings or for non-RAG queries.

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
    return {
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