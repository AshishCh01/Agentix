from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared state passed across nodes in the LangGraph multi-agent pipeline.
    """
    session_id: str
    user_id: str
    user_query: str
    standalone_query: Optional[str]  # <-- ADDED: Rewritten query with conversation context
    image_data: Optional[str]  # <-- ADDED: Base64 image payload
    intent: Optional[str]
    chat_history: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    formatted_context: str
    context_source: Optional[str]  # "document" | "web" — where formatted_context came from this pass
    tool_outputs: List[Dict[str, Any]]
    final_response: str
    error: Optional[str]
    retry_count: int