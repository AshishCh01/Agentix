import logging
from typing import Any, Dict
from ai.agents.state import AgentState
from ai.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)

WEB_ANSWER_SYSTEM_PROMPT = """You are an expert Q&A assistant grounded in live web search results.
Synthesize a clear, concise, and accurate answer using ONLY the provided web search context.

Web Search Context:
{context}
"""


async def run_answer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a final grounded response using retrieved context (RAG or Web)
    and session conversation history.
    """
    final_response = state.get("final_response", "")
    if final_response:
        return {"final_response": final_response}

    formatted_context = state.get("formatted_context", "").strip()
    intent = state.get("intent", "RAG_QUERY")

    # Fallback when no context is retrieved
    if not formatted_context:
        if intent == "WEB_SEARCH":
            return {
                "final_response": (
                    "I searched the web but could not retrieve live results "
                    "at this time. Please try rephrasing your search query."
                )
            }
        return {
            "final_response": (
                "I couldn't find any relevant information in your uploaded documents "
                "to answer your question. Please ensure the relevant document has been "
                "uploaded to this session."
            )
        }

    # Use web prompt for WEB_SEARCH, document prompt for RAG_QUERY
    if intent == "WEB_SEARCH":
        system_instruction = WEB_ANSWER_SYSTEM_PROMPT.format(context=formatted_context)
    else:
        system_instruction = ANSWER_SYSTEM_PROMPT.format(context=formatted_context)

    messages = [{"role": "system", "content": system_instruction}]

    chat_history = state.get("chat_history", [])
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_query = state.get("user_query", "")
    messages.append({"role": "user", "content": user_query})

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
            "final_response": "An error occurred while generating the answer from search results.",
        }