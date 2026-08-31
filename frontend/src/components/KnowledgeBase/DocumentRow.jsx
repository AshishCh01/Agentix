import { useState } from "react";
import { CheckCircle2, FileText, Loader2, TriangleAlert, Trash2 } from "lucide-react";
import { formatDate } from "../../utils/helpers";

const formatFileSize = (bytes) => {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const STATUS_BADGES = {
  processing: {
    icon: Loader2,
    iconClass: "animate-spin",
    className:
      "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/20",
    label: "Processing",
  },
  ready: {
    icon: CheckCircle2,
    iconClass: "",
    className:
      "bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20",
    label: "Ready",
  },
  failed: {
    icon: TriangleAlert,
    iconClass: "",
    className:
      "bg-red-50 text-red-700 border-red-100 dark:bg-red-500/10 dark:text-red-400 dark:border-red-500/20",
    label: "Failed",
  },
};

export const DocumentRow = ({ document, onDelete }) => {
  const [isDeleting, setIsDeleting] = useState(false);
  const badge = STATUS_BADGES[document.status] || STATUS_BADGES.processing;
  const BadgeIcon = badge.icon;

  const handleDelete = async () => {
    if (isDeleting) return;
    if (!window.confirm(`Delete "${document.filename}"?`)) return;
    setIsDeleting(true);
    try {
      await onDelete(document.id);
    } catch (err) {
      console.error("Failed to delete document:", err);
      setIsDeleting(false);
    }
  };

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-white border border-slate-200 dark:bg-slate-900 dark:border-slate-800 rounded-xl">
      <div className="flex items-center space-x-3 min-w-0 pr-3">
        <div className="p-2 bg-blue-50 border border-blue-100 text-blue-600 dark:bg-blue-600/10 dark:border-blue-500/20 dark:text-blue-400 rounded-lg flex-shrink-0">
          <FileText className="w-4 h-4" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">
            {document.filename}
          </p>
          <p className="text-[11px] text-slate-400 dark:text-slate-500">
            {formatFileSize(document.file_size)} &middot; Uploaded {formatDate(document.created_at)}
          </p>
          {document.status === "failed" && document.error_message && (
            <p className="text-[11px] text-red-500 dark:text-red-400 mt-0.5 truncate">
              {document.error_message}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center space-x-2 flex-shrink-0">
        <span
          className={`flex items-center gap-1 px-2 py-1 text-[10px] font-medium border rounded-full ${badge.className}`}
        >
          <BadgeIcon className={`w-3 h-3 ${badge.iconClass}`} />
          {badge.label}
        </span>
        <button
          onClick={handleDelete}
          disabled={isDeleting}
          title="Delete document"
          className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-slate-100 dark:text-slate-500 dark:hover:text-red-400 dark:hover:bg-slate-800 disabled:opacity-50 rounded-lg transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
