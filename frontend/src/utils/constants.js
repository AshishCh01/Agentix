export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

export const AGENT_NODES = {
  SUPERVISOR: "supervisor",
  VECTOR_SEARCH: "vector_search",
  WEB_SEARCH: "web_search",
  ANSWER: "answer",
  REFLECTION: "reflection",
  GREETING: "greeting",
};

export const AGENT_NODE_LABELS = {
  [AGENT_NODES.SUPERVISOR]: "Analyzing Intent",
  [AGENT_NODES.VECTOR_SEARCH]: "Searching Document Index",
  [AGENT_NODES.WEB_SEARCH]: "Browsing Live Web",
  [AGENT_NODES.ANSWER]: "Synthesizing Response",
  [AGENT_NODES.REFLECTION]: "Evaluating Response Quality",
  [AGENT_NODES.GREETING]: "Responding",
};

export const INTENTS = {
  RAG_QUERY: "RAG_QUERY",
  WEB_SEARCH: "WEB_SEARCH",
  GREETING: "GREETING",
};