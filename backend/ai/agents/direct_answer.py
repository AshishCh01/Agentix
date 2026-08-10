import logging
from typing import Any, Dict
from ai.agents.state import AgentState
from ai.prompts.direct_answer_prompt import DIRECT_ANSWER_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)


async def run_direct_answer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent for handling queries that don't need external data (math, coding, general facts).
    Generates a helpful reply directly.
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])

    messages = [{"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT}]

    # Include recent chat history
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    try:
        response = await llm_service.generate_response(
            messages=messages,
            temperature=0.4,
            max_tokens=1000,
        )
        return {
            "intent": "DIRECT_ANSWER",
            "final_response": response,
        }
    except Exception as e:
        logger.error(f"Direct answer agent failed: {str(e)}")
        return {
            "intent": "DIRECT_ANSWER",
            "error": f"Direct answer agent error: {str(e)}",
            "final_response": (
                "I encountered an error while trying to process your request directly."
            ),
        }
