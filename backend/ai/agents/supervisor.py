import logging
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field
from ai.agents.state import AgentState
from ai.prompts.supervisor_prompt import SUPERVISOR_PROMPT
from ai.services.llm_service import llm_service
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class IntentClassification(BaseModel):
    intent: Literal["GREETING", "RAG_QUERY", "WEB_SEARCH", "DIRECT_ANSWER"] = Field(
        description="The classified intent of the user query."
    )
    standalone_query: str = Field(
        description="The query rewritten to be self-contained using chat history. If already self-contained, this should be identical to the original user query."
    )

async def classify_intent(state: AgentState) -> Dict[str, Any]:
    """
    Classifies user intent into GREETING, RAG_QUERY, WEB_SEARCH, or DIRECT_ANSWER
    using native structured output from the LLM, and generates a standalone query.
    Returns a dict with 'intent' and 'standalone_query'.
    """
    user_query = state.get("user_query", "")
    image_data = state.get("image_data")
    chat_history = state.get("chat_history", [])

    # Fast-path heuristic for simple greetings to bypass LLM latency
    if not chat_history and not image_data:
        q_lower = user_query.strip().lower()
        if q_lower in ["hi", "hello", "hey", "hi there", "hello there", "greetings"]:
            logger.info("⚡ Fast-path greeting detected. Bypassing LLM supervisor.")
            return {
                "intent": "GREETING",
                "standalone_query": user_query
            }

    # Build Multimodal Content for LLM Classification
    user_content: List[Dict[str, Any]] = []
    
    if chat_history:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]])
        user_content.append({"type": "text", "text": f"Chat History:\n{history_str}\n\n"})

    if user_query.strip():
        user_content.append({"type": "text", "text": f"User Query: {user_query}"})
    else:
        user_content.append({"type": "text", "text": "User Query: [An image was attached without text]"})

    if image_data:
        image_url = image_data if image_data.startswith("data:") else f"data:image/png;base64,{image_data}"
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        llm = llm_service.get_chat_model(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(IntentClassification)
        result = await structured_llm.ainvoke(messages)
        return {
            "intent": result.intent,
            "standalone_query": result.standalone_query
        }
    except Exception as e:
        logger.warning(
            f"Intent classification structured output failed ({str(e)}). Applying fallback."
        )
        fallback_intent = "DIRECT_ANSWER"
        if image_data:
            fallback_intent = "RAG_QUERY"
        elif any(k in user_query.lower() for k in ["hi", "hello", "hey"]):
            fallback_intent = "GREETING"
            
        return {
            "intent": fallback_intent,
            "standalone_query": user_query
        }


async def run_supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph entry node wrapper for supervisor routing.
    """
    return await classify_intent(state)