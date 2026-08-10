import logging
from typing import Any, Dict
from ai.agents.state import AgentState
from ai.prompts.greeting_prompt import GREETING_SYSTEM_PROMPT
from ai.services.llm_service import llm_service

logger = logging.getLogger(__name__)


async def run_greeting_agent(state: AgentState) -> Dict[str, Any]:
    """
    Fast-path agent for handling user greetings and pleasantries.
    Generates a friendly reply without triggering vector database or web search tools.
    Uses ChatOpenAI via llm_service to enable streaming callback events in LangGraph.
    """
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])

    messages = [{"role": "system", "content": GREETING_SYSTEM_PROMPT}]

    # Include recent chat history if present
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    try:
        response = await llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=200,
        )
        return {
            "intent": "GREETING",
            "final_response": response,
        }
    except Exception as e:
        logger.error(f"Greeting agent failed: {str(e)}")
        return {
            "intent": "GREETING",
            "error": f"Greeting agent error: {str(e)}",
            "final_response": (
                "Hello! How can I assist you with your documents or queries today?"
            ),
        }