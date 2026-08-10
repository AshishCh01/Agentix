import logging
from typing import Any, Dict
from pydantic import BaseModel, Field

from ai.agents.state import AgentState
from ai.services.llm_service import llm_service
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are a Quality & Groundedness Evaluator for an AI Assistant.
Analyze the provided Assistant Response against the Context and User Query.

Evaluate two criteria:
1. Groundedness: Is the response factual and supported by the provided context or greeting?
2. Relevance: Does the response directly address the user's query?
"""

class ReflectionOutput(BaseModel):
    is_grounded: bool = Field(description="True if the response is supported by the provided context.")
    is_relevant: bool = Field(description="True if the response directly addresses the user's query.")
    reason: str = Field(description="Brief explanation of the evaluation result.")

async def evaluate_response(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the assistant's response for groundedness and relevance.
    Clears error state if response passes evaluation.
    """
    user_query = state.get("user_query", "")
    formatted_context = state.get("formatted_context", "")
    final_response = state.get("final_response", "")
    intent = state.get("intent", "RAG_QUERY")

    if intent == "GREETING" or not final_response:
        return {"error": None}

    logger.info("🔍 [Reflection Node] Starting response groundedness & relevance check...")

    eval_prompt = f"""
    Context:
    {formatted_context if formatted_context else "No context provided."}

    User Query:
    {user_query}

    Assistant Response:
    {final_response}
    """

    messages = [
        SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
        HumanMessage(content=eval_prompt),
    ]

    tool_outputs = state.get("tool_outputs", [])

    try:
        llm = llm_service.get_chat_model(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(ReflectionOutput)
        parsed = await structured_llm.ainvoke(messages)
        
        if not parsed.is_grounded or not parsed.is_relevant:
            logger.warning(
                f"⚠️ [Reflection Check Failed] Grounded: {parsed.is_grounded}, Relevant: {parsed.is_relevant}. Reason: {parsed.reason}"
            )
            return {
                "error": f"Reflection Check Failed: {parsed.reason}",
                "tool_outputs": tool_outputs + [
                    {"node": "reflection", "result": parsed.model_dump(), "is_grounded": parsed.is_grounded, "is_relevant": parsed.is_relevant}
                ],
            }

        logger.info("✅ [Reflection Node] Response verified successfully (Grounded & Relevant).")
        return {
            "error": None,
            "tool_outputs": tool_outputs + [
                {"node": "reflection", "result": parsed.model_dump(), "is_grounded": True, "is_relevant": True}
            ],
        }

    except Exception as e:
        logger.error(f"❌ [Reflection Node Error]: {str(e)}")
        # Strict fail state if evaluation crashes (don't default to True)
        return {
            "error": f"Evaluator Error: {str(e)}",
            "tool_outputs": tool_outputs + [
                {"node": "reflection", "result": {"is_grounded": False, "is_relevant": False, "reason": "Evaluator failed."}}
            ],
        }


async def run_reflection_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node alias for reflection execution.
    """
    return await evaluate_response(state)