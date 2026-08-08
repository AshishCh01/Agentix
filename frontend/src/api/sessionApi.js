import apiClient from "./axios";

export const sessionApi = {
  getSessions: async () => {
    const response = await apiClient.get("/sessions");
    return response.data;
  },

  createSession: async (title = "New Chat Session") => {
    const response = await apiClient.post("/sessions", { title });
    return response.data;
  },

  getSessionMessages: async (sessionId) => {
    const response = await apiClient.get(`/sessions/${sessionId}/messages`);
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await apiClient.delete(`/sessions/${sessionId}`);
    return response.data;
  },
};