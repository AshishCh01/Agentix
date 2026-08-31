import apiClient from "./axios";

export const knowledgeBaseApi = {
  getKnowledgeBases: async () => {
    const response = await apiClient.get("/knowledge-bases");
    return response.data;
  },

  getKnowledgeBase: async (kbId) => {
    const response = await apiClient.get(`/knowledge-bases/${kbId}`);
    return response.data;
  },

  createKnowledgeBase: async (name, description) => {
    const response = await apiClient.post("/knowledge-bases", { name, description });
    return response.data;
  },

  updateKnowledgeBase: async (kbId, { name, description } = {}) => {
    const payload = {};
    if (name !== undefined) payload.name = name;
    if (description !== undefined) payload.description = description;
    const response = await apiClient.patch(`/knowledge-bases/${kbId}`, payload);
    return response.data;
  },

  deleteKnowledgeBase: async (kbId) => {
    const response = await apiClient.delete(`/knowledge-bases/${kbId}`);
    return response.data;
  },

  getDocuments: async (kbId) => {
    const response = await apiClient.get(`/knowledge-bases/${kbId}/documents`);
    return response.data.documents;
  },

  uploadDocument: async (kbId, file, onUploadProgress) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await apiClient.post(`/knowledge-bases/${kbId}/documents`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progressEvent) => {
        if (onUploadProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onUploadProgress(percentCompleted);
        }
      },
    });

    return response.data;
  },

  deleteDocument: async (kbId, documentId) => {
    const response = await apiClient.delete(`/knowledge-bases/${kbId}/documents/${documentId}`);
    return response.data;
  },
};
