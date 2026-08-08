import logging
from typing import Any, Dict
from ai.agents.state import AgentState
from ai.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)


async def run_answer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a final grounded response using retrieved vector context and
    session conversation history.
    """
    final_response = state.get("final_response", "")
    if final_response:
        return {"final_response": final_response}

    formatted_context = state.get("formatted_context", "")

    # Fallback when no context chunks are retrieved
    if not formatted_context.strip():
        return {
            "final_response": (
                "I couldn't find any relevant information in your uploaded documents "
                "to answer your question. Please ensure the relevant document has been "
                "uploaded to this session."
            )
        }

    # Format system prompt with retrieved context
    system_instruction = ANSWER_SYSTEM_PROMPT.format(context=formatted_context)
    messages = [{"role": "system", "content": system_instruction}]

    chat_history = state.get("chat_history", [])
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_query = state.get("user_query", "")
    messages.append({"role": "user", "content": user_query})

    # Generate grounded response using low temperature to minimize hallucinations
    try:
        response = await llm_service.generate_response(
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        return {"final_response": response}
    except Exception as e:
        logger.error(f"Answer synthesis failed: {str(e)}")
        return {
            "error": f"Answer agent error: {str(e)}",
            "final_response": (
                "An error occurred while generating the answer from your document context."
            ),
        }