import { FileText, CheckCircle2, AlertCircle, X } from "lucide-react";
import { Spinner } from "../Common/Loader";

export const FilePreview = ({ file, progress, isUploading, error, success, onClear }) => {
  if (!file) return null;

  const formatFileSize = (bytes) => {
    if (!bytes) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  return (
    <div className="p-3 bg-slate-50 border border-slate-200 dark:bg-slate-900 dark:border-slate-800 rounded-xl space-y-2 text-xs">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5 min-w-0 pr-2">
          <div className="p-2 bg-blue-50 border border-blue-100 text-blue-600 dark:bg-blue-600/10 dark:border-blue-500/20 dark:text-blue-400 rounded-lg flex-shrink-0">
            <FileText className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <p className="font-medium text-slate-700 dark:text-slate-200 truncate">{file.name}</p>
            <p className="text-[10px] text-slate-400 dark:text-slate-500">{formatFileSize(file.size)}</p>
          </div>
        </div>

        <button
          onClick={onClear}
          disabled={isUploading}
          className="p-1 text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300 disabled:opacity-50 rounded"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Uploading Progress */}
      {isUploading && (
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
            <span className="flex items-center gap-1.5">
              <Spinner size="sm" />
              Transferring file to server...
            </span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-blue-500 h-full transition-all duration-300 rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Success Badge - Updated to reflect 202 Accepted status accurately */}
      {success && (
        <div className="flex items-center space-x-1.5 text-emerald-600 dark:text-emerald-400 font-medium pt-1">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>Upload complete! Processing in background.</span>
        </div>
      )}

      {/* Error Badge */}
      {error && (
        <div className="flex items-center space-x-1.5 text-red-600 dark:text-red-400 font-medium pt-1">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}
    </div>
  );
};