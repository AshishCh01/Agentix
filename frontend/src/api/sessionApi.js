import apiClient from "./axios";

export const sessionApi = {
  getSessions: async () => {
    const response = await apiClient.get("/sessions");
    return response.data;
  },

  createSession: async (title = "New Chat Session", knowledgeBaseId = null) => {
    const payload = { title };
    if (knowledgeBaseId) payload.knowledge_base_id = knowledgeBaseId;
    const response = await apiClient.post("/sessions", payload);
    return response.data;
  },

  getSessionMessages: async (sessionId) => {
    const response = await apiClient.get(`/sessions/${sessionId}/messages`);
    
    // Normalize 'sender' to 'role' to fix chat history display alignment
    return response.data.map(msg => ({
      ...msg,
      role: msg.sender || msg.role
    }));
  },

  deleteSession: async (sessionId) => {
    const response = await apiClient.delete(`/sessions/${sessionId}`);
    return response.data;
  },
};