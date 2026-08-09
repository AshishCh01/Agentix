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
from ai.tools.vector_search import vector_search_tool
from ai.tools.web_search import web_search_tool
from app.config.settings import settings

logger = logging.getLogger(__name__)


# --- 1. Node Definitions ---

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
    # 1. Global toggle check from settings / .env
    if getattr(settings, "SKIP_REFLECTION", False):
        logger.info("⏩ Skipping reflection node (SKIP_REFLECTION=True).")
        return END

    # 2. Skip reflection for Web Search and Greetings (hallucination checks not needed)
    intent = state.get("intent")
    if intent in ["WEB_SEARCH", "GREETING"]:
        logger.info(f"⏩ Skipping reflection node for intent: {intent}.")
        return END

    # 3. Only execute reflection for private document RAG queries
    return "reflection"


async def greeting_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fast-path greeting generator node."""
    return await run_greeting_agent(state)


async def vector_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Vector database retrieval node."""
    configurable = config.get("configurable", {}) if config else {}
    db = configurable.get("db")

    if db is None:
        raise ValueError("AsyncSession 'db' was not provided in RunnableConfig['configurable']")

    session_uuid = (
        state["session_id"]
        if isinstance(state["session_id"], uuid.UUID)
        else uuid.UUID(state["session_id"])
    )

    tool_result = await vector_search_tool(
        db=db,
        session_id=session_uuid,
        query=state["user_query"],
        image_data=state.get("image_data"),  # <-- PASSING IMAGE DATA FOR VISUAL SEARCH
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
    return await run_answer_agent(state)


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

    # Conditional Routing Edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "greeting": "greeting",
            "vector_search": "vector_search",
            "web_search": "web_search",
        },
    )

    # Conditional Routing Edge after Vector Search (Fallback to Web Search)
    workflow.add_conditional_edges(
        "vector_search",
        route_after_vector_search,
        {
            "web_search": "web_search",
            "answer": "answer",
        },
    )

    # Conditional Routing Edge after Answer Node (Conditional Reflection)
    workflow.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "reflection": "reflection",
            END: END,
        },
    )

    # Direct Node Edges
    workflow.add_edge("greeting", END)
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("reflection", END)

    return workflow.compile()


# Compiled Graph Instance
rag_graph = build_graph()