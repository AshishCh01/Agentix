import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from ai.agents.state import AgentState
from ai.services.llm_service import llm_service
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

UNVERIFIED_ANSWER_CAVEAT = (
    "\n\n> ⚠️ **Note:** This response could not be fully verified against your uploaded "
    "documents and may contain incomplete or ungrounded details."
)

REFLECTION_SYSTEM_PROMPT = """You are a Quality & Groundedness Evaluator for an AI Assistant.
Analyze the provided Assistant Response against the Context, User Query, and any Chat History.

Evaluate two criteria:
1. Groundedness: Is the response factual and supported by the provided context or greeting?
2. Relevance: Does the response directly address the user's query, taking into account recent Chat History for follow-ups and conversational context?
"""


class ReflectionOutput(BaseModel):
    is_grounded: bool = Field(description="True if the response is supported by the provided context.")
    is_relevant: bool = Field(description="True if the response directly addresses the user's query.")
    reason: str = Field(description="Brief explanation of the evaluation result.")


async def evaluate_response(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the assistant's response for groundedness and relevance in the context
    of the retrieved documents, user query, and recent chat history.

    - If the response passes, error is cleared, verified=True, and response text is untouched.
    - If the retry cap is reached and the response is rejected, verified=False and an explicit,
      user-visible caveat is appended to the final_response.
    """
    user_query = state.get("user_query", "")
    formatted_context = state.get("formatted_context", "")
    final_response = state.get("final_response", "")
    intent = state.get("intent", "RAG_QUERY")
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    is_cap_reached = retry_count >= max_retries

    if intent == "GREETING" or not final_response:
        return {"error": None, "verified": True, "reflection_verdict": "passed"}

    # Guard: if context is completely empty and no image (e.g. not-covered fallback), skip reflection
    if not formatted_context and not state.get("image_data"):
        return {"error": None, "verified": False, "reflection_verdict": "not_applicable"}

    logger.info("🔍 [Reflection Node] Starting response groundedness & relevance check...")

    # Bounded chat history (last 6 turns, capped at ~2000 chars)
    chat_history: List[Dict[str, Any]] = state.get("chat_history", [])
    history_str = ""
    if chat_history:
        recent_turns = chat_history[-6:]
        formatted_turns = [
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent_turns
        ]
        history_str = "\n".join(formatted_turns)
        if len(history_str) > 2000:
            history_str = history_str[-2000:]

    eval_prompt = f"""
Context:
{formatted_context if formatted_context else "No context provided."}
"""
    if history_str:
        eval_prompt += f"""
Recent Chat History:
{history_str}
"""
    eval_prompt += f"""
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
                f"⚠️ [Reflection Check Failed] Grounded: {parsed.is_grounded}, "
                f"Relevant: {parsed.is_relevant}. Reason: {parsed.reason}"
            )

            unverified_response = final_response
            if is_cap_reached and UNVERIFIED_ANSWER_CAVEAT not in final_response:
                unverified_response = f"{final_response.rstrip()}{UNVERIFIED_ANSWER_CAVEAT}"

            return {
                "error": f"Reflection Check Failed: {parsed.reason}",
                "final_response": unverified_response,
                "verified": False,
                "reflection_verdict": "failed",
                "tool_outputs": tool_outputs + [
                    {
                        "node": "reflection",
                        "result": parsed.model_dump(),
                        "is_grounded": parsed.is_grounded,
                        "is_relevant": parsed.is_relevant,
                        "verified": False,
                    }
                ],
            }

        logger.info("✅ [Reflection Node] Response verified successfully (Grounded & Relevant).")
        return {
            "error": None,
            "verified": True,
            "reflection_verdict": "passed",
            "tool_outputs": tool_outputs + [
                {
                    "node": "reflection",
                    "result": parsed.model_dump(),
                    "is_grounded": True,
                    "is_relevant": True,
                    "verified": True,
                }
            ],
        }

    except Exception as e:
        logger.error(f"❌ [Reflection Node Error]: {str(e)}")
        unverified_response = final_response
        if is_cap_reached and UNVERIFIED_ANSWER_CAVEAT not in final_response:
            unverified_response = f"{final_response.rstrip()}{UNVERIFIED_ANSWER_CAVEAT}"

        return {
            "error": f"Evaluator Error: {str(e)}",
            "final_response": unverified_response,
            "verified": False,
            "reflection_verdict": "failed",
            "tool_outputs": tool_outputs + [
                {
                    "node": "reflection",
                    "result": {
                        "is_grounded": False,
                        "is_relevant": False,
                        "reason": f"Evaluator failed: {str(e)}",
                    },
                    "verified": False,
                }
            ],
        }


async def run_reflection_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node alias for reflection execution.
    """
    return await evaluate_response(state)
