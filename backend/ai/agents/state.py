from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared state passed across nodes in the LangGraph multi-agent pipeline.
    """
    session_id: str
    user_id: str
    user_query: str
    image_data: Optional[str]  # <-- ADDED: Base64 image payload
    intent: Optional[str]
    chat_history: List[Dict[str, str]]
    retrieved_chunks: List[Dict[str, Any]]
    formatted_context: str
    tool_outputs: List[Dict[str, Any]]
    final_response: str
    error: Optional[str]