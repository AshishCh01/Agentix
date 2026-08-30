export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

// Matches backend MAX_UPLOAD_SIZE_MB (app/config/settings.py) -- the single
// limit the backend enforces for both document uploads (/upload) and chat
// image attachments (/chat, /chat/stream). Checked client-side so an
// oversized file is rejected immediately instead of failing after a full
// request round-trip.
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

// Max number of past messages allowed to keep their full base64 image_data
// in memory at once. Older attachments are dropped as new ones arrive so a
// long-running chat session doesn't accumulate unbounded base64 payloads in
// React state.
export const MAX_RETAINED_MESSAGE_IMAGES = 5;

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