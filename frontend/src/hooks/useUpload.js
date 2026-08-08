import { useState } from "react";
import { uploadApi } from "../api/uploadApi";

export const useUpload = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const uploadFile = async (sessionId, file) => {
    if (!file || !sessionId) return null;

    setIsUploading(true);
    setProgress(0);
    setError(null);
    setSuccess(false);

    try {
      const response = await uploadApi.uploadDocument(
        sessionId,
        file,
        (percent) => setProgress(percent)
      );
      setSuccess(true);
      return response;
    } catch (err) {
      const msg = err?.response?.data?.detail || "Document upload failed.";
      setError(msg);
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  const resetUploadState = () => {
    setIsUploading(false);
    setProgress(0);
    setError(null);
    setSuccess(false);
  };

  return {
    uploadFile,
    isUploading,
    progress,
    error,
    success,
    resetUploadState,
  };
};