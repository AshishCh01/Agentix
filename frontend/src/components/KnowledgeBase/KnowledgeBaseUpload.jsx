import { useState, useRef } from "react";
import { FileUp } from "lucide-react";
import { FilePreview } from "../Upload/FilePreview";
import { MAX_UPLOAD_BYTES } from "../../utils/constants";
import { getErrorMessage } from "../../utils/helpers";

const MAX_UPLOAD_MB = Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024));
const VALID_EXTENSIONS = ["pdf", "docx", "csv", "xlsx", "txt", "md", "json"];

const isValidFile = (file) => {
  const ext = file.name.split(".").pop().toLowerCase();
  const isImage = file.type.startsWith("image/");
  return isImage || VALID_EXTENSIONS.includes(ext);
};

// Upload dropzone scoped to a single knowledge base. Mirrors
// components/Upload/UploadButton's inline flow (drag/drop, validation,
// FilePreview progress/success/error states) but uploads directly to the
// KB's document endpoint instead of a chat session, and calls back into the
// parent to refresh the document list on success.
export const KnowledgeBaseUpload = ({ onUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef(null);

  const resetState = () => {
    setIsUploading(false);
    setProgress(0);
    setError(null);
    setSuccess(false);
  };

  const validateAndSet = (file) => {
    if (!file) return;
    if (!isValidFile(file)) {
      alert("Unsupported file type. Please upload a PDF, Word Doc, CSV, Excel, TXT, or Image.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      alert(`File is too large (max ${MAX_UPLOAD_MB}MB). Please choose a smaller file.`);
      return;
    }
    setSelectedFile(file);
    resetState();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    validateAndSet(e.dataTransfer.files?.[0]);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);
    try {
      await onUpload(selectedFile, (percent) => setProgress(percent));
      setSuccess(true);
    } catch (err) {
      setError(getErrorMessage(err) || "Document upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    resetState();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="space-y-3">
      {!selectedFile && (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            isDragging
              ? "border-blue-500 bg-blue-50 dark:bg-blue-500/10"
              : "border-slate-300 hover:border-slate-400 bg-slate-50/50 dark:border-slate-800 dark:hover:border-slate-700 dark:bg-slate-950/50"
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => validateAndSet(e.target.files?.[0])}
            accept=".pdf,.docx,.csv,.xlsx,.txt,.md,.json,image/*"
            className="hidden"
          />
          <FileUp className="w-8 h-8 text-slate-400 dark:text-slate-500 mx-auto mb-2" />
          <p className="text-xs font-medium text-slate-600 dark:text-slate-300">
            Click to upload or drag & drop a file into this knowledge base
          </p>
          <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
            Supported: PDF, DOCX, CSV, Excel, TXT, Images (Max {MAX_UPLOAD_MB}MB)
          </p>
        </div>
      )}

      {selectedFile && (
        <>
          <FilePreview
            file={selectedFile}
            progress={progress}
            isUploading={isUploading}
            error={error}
            success={success}
            onClear={handleClear}
          />
          <div className="flex items-center justify-end space-x-2">
            {!success && (
              <button
                onClick={handleUploadSubmit}
                disabled={isUploading}
                className="px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition-colors"
              >
                {isUploading ? "Uploading..." : "Start Upload"}
              </button>
            )}
            {success && (
              <button
                onClick={handleClear}
                className="px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
              >
                Upload Another
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
};
