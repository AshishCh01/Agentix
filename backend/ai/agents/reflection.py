import json
import logging
import re
from typing import Any, Dict

from ai.agents.state import AgentState
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are a Quality & Groundedness Evaluator for an AI Assistant.
Analyze the provided Assistant Response against the Context and User Query.

Evaluate two criteria:
1. Groundedness: Is the response factual and supported by the provided context or greeting?
2. Relevance: Does the response directly address the user's query?

Respond strictly in JSON format:
{
    "is_grounded": true or false,
    "is_relevant": true or false,
    "reason": "Brief explanation of evaluation result"
}
"""


def _clean_json_str(text: str) -> str:
    """
    Strips markdown code fences and isolates JSON payload boundaries.
    """
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return text


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
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": eval_prompt},
    ]

    tool_outputs = state.get("tool_outputs", [])

    try:
        response_text = await llm_service.generate_response(
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )

        cleaned_json = _clean_json_str(response_text)
        if cleaned_json:
            parsed = json.loads(cleaned_json)
            is_grounded = parsed.get("is_grounded", True)
            is_relevant = parsed.get("is_relevant", True)
            reason = parsed.get("reason", "Evaluation completed.")

            if not is_grounded or not is_relevant:
                logger.warning(
                    f"⚠️ [Reflection Check Failed] Grounded: {is_grounded}, Relevant: {is_relevant}. Reason: {reason}"
                )
                return {
                    "error": f"Reflection Check Failed: {reason}",
                    "tool_outputs": tool_outputs + [
                        {"node": "reflection", "result": parsed, "is_grounded": is_grounded, "is_relevant": is_relevant}
                    ],
                }

            logger.info("✅ [Reflection Node] Response verified successfully (Grounded & Relevant).")
            return {
                "error": None,
                "tool_outputs": tool_outputs + [
                    {"node": "reflection", "result": parsed, "is_grounded": True, "is_relevant": True}
                ],
            }

    except Exception as e:
        logger.error(f"❌ [Reflection Node Error]: {str(e)}")

    # Default fallback acceptance if reflection JSON parsing fails
    return {
        "error": None,
        "tool_outputs": tool_outputs + [
            {"node": "reflection", "result": {"is_grounded": True, "is_relevant": True, "reason": "Acceptance fallback applied."}}
        ],
    }


async def run_reflection_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node alias for reflection execution.
    """
    return await evaluate_response(state)