import { useState } from "react";
import { createPortal } from "react-dom";
import { BookMarked, X } from "lucide-react";
import { Spinner } from "../Common/Loader";
import { getErrorMessage } from "../../utils/helpers";

// Shared create/rename modal for knowledge bases -- portaled to <body> to
// stay consistent with UploadButton's modal (escapes any backdrop-blur
// ancestor that would otherwise become its containing block).
export const KnowledgeBaseFormModal = ({ mode = "create", initialValues, onSubmit, onClose }) => {
  const [name, setName] = useState(initialValues?.name || "");
  const [description, setDescription] = useState(initialValues?.description || "");
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);

  const isEdit = mode === "edit";

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim() || isSaving) return;
    setIsSaving(true);
    setError(null);
    try {
      await onSubmit({ name: name.trim(), description: description.trim() || null });
      onClose();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsSaving(false);
    }
  };

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 dark:bg-slate-950/80 backdrop-blur-sm overflow-y-auto">
      <div className="w-full max-w-md my-auto bg-white border border-slate-200 dark:bg-slate-900 dark:border-slate-800 rounded-2xl p-5 shadow-2xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <BookMarked className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">
              {isEdit ? "Rename Knowledge Base" : "New Knowledge Base"}
            </h3>
          </div>
          <button
            onClick={onClose}
            disabled={isSaving}
            className="p-1 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 disabled:opacity-50"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Name
            </label>
            <input
              autoFocus
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={255}
              placeholder="e.g. Product Documentation"
              className="w-full px-3 py-2.5 text-sm bg-slate-50 border border-slate-200 dark:bg-slate-950 dark:border-slate-800 dark:text-slate-100 rounded-xl focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/50 transition-all"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">
              Description <span className="text-slate-400 dark:text-slate-500">(optional)</span>
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={2000}
              rows={3}
              placeholder="What kind of documents live here?"
              className="w-full px-3 py-2.5 text-sm bg-slate-50 border border-slate-200 dark:bg-slate-950 dark:border-slate-800 dark:text-slate-100 rounded-xl focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/50 transition-all resize-none"
            />
          </div>

          {error && (
            <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
          )}

          <div className="flex items-center justify-end space-x-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              disabled={isSaving}
              className="px-3.5 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 dark:text-slate-400 dark:hover:text-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || isSaving}
              className="px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition-colors flex items-center space-x-1.5"
            >
              {isSaving && <Spinner size="sm" className="text-white" />}
              <span>{isEdit ? "Save Changes" : "Create Knowledge Base"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
};
