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

    try:
        response_text = await llm_service.generate_response(
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            is_grounded = parsed.get("is_grounded", True)
            is_relevant = parsed.get("is_relevant", True)
            reason = parsed.get("reason", "")

            if not is_grounded or not is_relevant:
                logger.warning(
                    f"⚠️ [Reflection Check Failed] Grounded: {is_grounded}, Relevant: {is_relevant}. Reason: {reason}"
                )
                return {
                    "error": f"Reflection Check Failed: {reason}",
                    "tool_outputs": state.get("tool_outputs", []) + [
                        {"tool": "reflection", "result": parsed}
                    ],
                }

        logger.info("✅ [Reflection Node] Response verified successfully (Grounded & Relevant).")
        return {
            "error": None,
            "tool_outputs": state.get("tool_outputs", []) + [
                {"tool": "reflection", "result": {"is_grounded": True, "is_relevant": True}}
            ],
        }
    except Exception as e:
        logger.error(f"❌ [Reflection Node Error]: {str(e)}")
        return {"error": None, "tool_outputs": state.get("tool_outputs", [])}