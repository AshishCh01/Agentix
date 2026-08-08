import logging
import uuid
from typing import Any, Dict
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from ai.agents.answer import run_answer_agent
from ai.agents.greeting import run_greeting_agent
from ai.agents.state import AgentState
from ai.agents.supervisor import classify_intent
from ai.tools.vector_search import vector_search_tool

logger = logging.getLogger(__name__)


# --- 1. Node Definitions ---

async def supervisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Classifies user intent (GREETING vs RAG_QUERY)."""
    intent = await classify_intent(state)
    return {"intent": intent}


def route_intent(state: AgentState) -> str:
    """Conditional Edge router function."""
    intent = state.get("intent", "RAG_QUERY")
    if intent == "GREETING":
        return "greeting"
    return "vector_search"


async def greeting_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Fast-path greeting generator node."""
    updated = await run_greeting_agent(state)
    return {"final_response": updated.get("final_response", "")}


async def vector_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Vector database retrieval node."""
    configurable = config.get("configurable", {}) if config else {}
    db = configurable.get("db")

    if db is None:
        raise ValueError("AsyncSession 'db' was not provided in RunnableConfig['configurable']")

    # Convert session_id string to UUID object
    session_uuid = (
        state["session_id"]
        if isinstance(state["session_id"], uuid.UUID)
        else uuid.UUID(state["session_id"])
    )

    tool_result = await vector_search_tool(
        db=db,
        session_id=session_uuid,
        query=state["user_query"],
        top_k=4,
    )
    return {
        "retrieved_chunks": tool_result.get("chunks", []),
        "formatted_context": tool_result.get("context_text", ""),
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "vector_search", "result": tool_result}],
    }


async def answer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Answer synthesis agent node."""
    updated = await run_answer_agent(state)
    return {"final_response": updated.get("final_response", "")}


# --- 2. Build StateGraph Workflow ---

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("answer", answer_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Conditional Routing Edges
    workflow.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "greeting": "greeting",
            "vector_search": "vector_search",
        },
    )

    # Node Edges
    workflow.add_edge("greeting", END)
    workflow.add_edge("vector_search", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


# Compiled Graph Instance
rag_graph = build_graph()