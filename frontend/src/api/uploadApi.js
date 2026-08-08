import apiClient from "./axios";

export const uploadApi = {
  uploadDocument: async (sessionId, file, onUploadProgress) => {
    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("file", file);

    const response = await apiClient.post("/upload", formData, {
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
};