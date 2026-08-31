import { useState, useRef } from "react";
import { createPortal } from "react-dom";
import { Upload, FileUp, X } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { useUpload } from "../../hooks/useUpload";
import { FilePreview } from "./FilePreview";
import { MAX_UPLOAD_BYTES } from "../../utils/constants";

const MAX_UPLOAD_MB = Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024));

export const UploadButton = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const { activeSessionId, createNewSession } = useChat();
  const { uploadFile, isUploading, progress, error, success, resetUploadState } = useUpload();

  // Helper to validate file types supported by the backend
  const isValidFile = (file) => {
    const validExtensions = ['pdf', 'docx', 'csv', 'xlsx', 'txt', 'md', 'json'];
    const ext = file.name.split('.').pop().toLowerCase();
    const isImage = file.type.startsWith('image/');
    return isImage || validExtensions.includes(ext);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!isValidFile(file)) {
        alert("Unsupported file type. Please upload a PDF, Word Doc, CSV, Excel, TXT, or Image.");
        e.target.value = "";
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        alert(`File is too large (max ${MAX_UPLOAD_MB}MB). Please choose a smaller file.`);
        e.target.value = "";
        return;
      }
      setSelectedFile(file);
      resetUploadState();
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (!isValidFile(file)) {
        alert("Unsupported file type. Please drop a valid document or image.");
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        alert(`File is too large (max ${MAX_UPLOAD_MB}MB). Please choose a smaller file.`);
        return;
      }
      setSelectedFile(file);
      resetUploadState();
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;

    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      const created = await createNewSession(`Doc: ${selectedFile.name.slice(0, 20)}`);
      targetSessionId = created.id;
    }

    try {
      await uploadFile(targetSessionId, selectedFile);
    } catch (err) {
      console.error("Failed to upload document:", err);
    }
  };

  const handleClose = () => {
    if (isUploading) return;
    setIsOpen(false);
    setSelectedFile(null);
    resetUploadState();
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center space-x-2 px-3 py-2 bg-slate-100 hover:bg-slate-200/80 border border-slate-200 text-slate-700 dark:bg-slate-800 dark:hover:bg-slate-700/80 dark:border-slate-700/50 dark:text-slate-200 text-xs font-medium rounded-lg transition-colors"
      >
        <Upload className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
        <span>Upload Document</span>
      </button>

      {/* Modal Dialog — portaled to <body> so it escapes the Navbar's
          backdrop-blur, which otherwise becomes the containing block for
          this fixed-position overlay and confines it to the navbar's box */}
      {isOpen && createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 dark:bg-slate-950/80 backdrop-blur-sm overflow-y-auto">
          <div className="w-full max-w-md my-auto max-h-[calc(100vh-2rem)] overflow-y-auto custom-scrollbar bg-white border border-slate-200 dark:bg-slate-900 dark:border-slate-800 rounded-2xl p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <FileUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">Upload Document</h3>
              </div>
              <button
                onClick={handleClose}
                disabled={isUploading}
                className="p-1 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 disabled:opacity-50"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Drag and Drop Zone */}
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
                  onChange={handleFileChange}
                  accept=".pdf,.docx,.csv,.xlsx,.txt,.md,.json,image/*"
                  className="hidden"
                />
                <FileUp className="w-8 h-8 text-slate-400 dark:text-slate-500 mx-auto mb-2" />
                <p className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Click to upload or drag & drop a file
                </p>
                <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                  Supported: PDF, DOCX, CSV, Excel, TXT, Images (Max {MAX_UPLOAD_MB}MB)
                </p>
              </div>
            )}

            {/* Selected File Preview Component */}
            {selectedFile && (
              <FilePreview
                file={selectedFile}
                progress={progress}
                isUploading={isUploading}
                error={error}
                success={success}
                onClear={() => {
                  setSelectedFile(null);
                  resetUploadState();
                }}
              />
            )}

            {/* Actions */}
            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                onClick={handleClose}
                disabled={isUploading}
                className="px-3.5 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 dark:text-slate-400 dark:hover:text-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 rounded-lg transition-colors"
              >
                {success ? "Close" : "Cancel"}
              </button>

              {selectedFile && !success && (
                <button
                  onClick={handleUploadSubmit}
                  disabled={isUploading}
                  className="px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition-colors flex items-center space-x-1.5"
                >
                  <span>{isUploading ? "Uploading..." : "Start Upload"}</span>
                </button>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
};