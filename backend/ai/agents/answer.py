import logging
from typing import Any, Dict, List
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
    Synthesizes a final grounded response using retrieved context (RAG or Web),
    session conversation history, and optional user image input.
    Uses ChatOpenAI via llm_service to enable streaming callback events in LangGraph.
    """
    # Remove the caching block that was returning the old answer on retry!
    
    formatted_context = state.get("formatted_context", "").strip()
    intent = state.get("intent", "RAG_QUERY")
    image_data = state.get("image_data")

    # Fallback when no context is retrieved AND no image is provided
    if not formatted_context and not image_data:
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

    # Select prompt template based on intent
    if intent == "WEB_SEARCH":
        system_instruction = WEB_ANSWER_SYSTEM_PROMPT.format(context=formatted_context)
    else:
        system_instruction = ANSWER_SYSTEM_PROMPT.format(
            context=formatted_context if formatted_context else "No document chunks retrieved."
        )
        
    error_feedback = state.get("error")
    if error_feedback:
        logger.info(f"🔄 Injecting reflection feedback into answer generation: {error_feedback}")
        system_instruction += f"\n\nCRITICAL FEEDBACK ON PREVIOUS ATTEMPT:\n{error_feedback}\nPlease correct your response based on this feedback."

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]

    # Append recent chat history
    chat_history = state.get("chat_history", [])
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Build final user query message (Text + Optional Image)
    user_query = state.get("user_query", "").strip()
    
    if image_data:
        image_url = (
            image_data if image_data.startswith("data:") else f"data:image/png;base64,{image_data}"
        )
        prompt_text = user_query if user_query else "Please analyze this image alongside the provided session context."
        
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_query})

    try:
        response = await llm_service.generate_response(
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        return {"final_response": response}
    except Exception as e:
        logger.error(f"Answer synthesis failed: {str(e)}")
        return {
            "error": f"Answer agent error: {str(e)}",
            "final_response": "An error occurred while generating the answer from search results or image payload.",
        }