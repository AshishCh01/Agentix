import { useState } from "react";
import { Link } from "react-router-dom";
import { BookMarked, FileText, Pencil, Trash2 } from "lucide-react";
import { formatDate, truncateText } from "../../utils/helpers";

export const KnowledgeBaseCard = ({ kb, onRename, onDelete }) => {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (isDeleting) return;
    if (!window.confirm(`Delete "${kb.name}"? This removes all of its documents too.`)) return;
    setIsDeleting(true);
    try {
      await onDelete(kb.id);
    } catch (err) {
      console.error("Failed to delete knowledge base:", err);
      setIsDeleting(false);
    }
  };

  const handleRename = (e) => {
    e.preventDefault();
    e.stopPropagation();
    onRename(kb);
  };

  return (
    <Link
      to={`/knowledge-bases/${kb.id}`}
      className="group block p-4 bg-white border border-slate-200 hover:border-blue-300 dark:bg-slate-900 dark:border-slate-800 dark:hover:border-blue-500/50 rounded-2xl shadow-sm hover:shadow-md transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="p-2.5 bg-blue-50 border border-blue-100 text-blue-600 dark:bg-blue-600/10 dark:border-blue-500/20 dark:text-blue-400 rounded-xl">
          <BookMarked className="w-5 h-5" />
        </div>

        <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleRename}
            title="Rename"
            className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-slate-100 dark:text-slate-500 dark:hover:text-blue-400 dark:hover:bg-slate-800 rounded-lg transition-colors"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            title="Delete"
            className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-slate-100 dark:text-slate-500 dark:hover:text-red-400 dark:hover:bg-slate-800 disabled:opacity-50 rounded-lg transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <h3 className="mt-3 text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
        {kb.name}
      </h3>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 line-clamp-2 min-h-[2rem]">
        {kb.description ? truncateText(kb.description, 90) : "No description"}
      </p>

      <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 dark:text-slate-500">
        <span className="flex items-center gap-1">
          <FileText className="w-3.5 h-3.5" />
          {kb.document_count} {kb.document_count === 1 ? "document" : "documents"}
        </span>
        <span>Updated {formatDate(kb.updated_at)}</span>
      </div>
    </Link>
  );
};
