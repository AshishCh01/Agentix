import { useState } from "react";
import { uploadApi } from "../api/uploadApi";
import { MAX_UPLOAD_BYTES } from "../utils/constants";

export const useUpload = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const uploadFile = async (sessionId, file) => {
    if (!file || !sessionId) return null;

    // Reject oversized files before ever hitting the network -- the
    // backend enforces the same MAX_UPLOAD_SIZE_MB limit and would 413 the
    // request anyway, after the user waited through an upload attempt.
    if (file.size > MAX_UPLOAD_BYTES) {
      const msg = `File is too large (max ${Math.floor(
        MAX_UPLOAD_BYTES / (1024 * 1024)
      )}MB).`;
      setError(msg);
      setSuccess(false);
      throw new Error(msg);
    }

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