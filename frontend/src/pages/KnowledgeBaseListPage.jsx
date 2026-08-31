import { useState } from "react";
import { BookMarked, Plus, TriangleAlert } from "lucide-react";
import { useKnowledgeBase } from "../hooks/useKnowledgeBase";
import { Spinner } from "../components/Common/Loader";
import { KnowledgeBaseCard } from "../components/KnowledgeBase/KnowledgeBaseCard";
import { KnowledgeBaseFormModal } from "../components/KnowledgeBase/KnowledgeBaseFormModal";

export const KnowledgeBaseListPage = () => {
  const {
    knowledgeBases,
    loading,
    error,
    fetchKnowledgeBases,
    createKnowledgeBase,
    updateKnowledgeBase,
    deleteKnowledgeBase,
  } = useKnowledgeBase();

  const [modalMode, setModalMode] = useState(null); // null | "create" | "edit"
  const [editingKb, setEditingKb] = useState(null);

  const openCreateModal = () => {
    setEditingKb(null);
    setModalMode("create");
  };

  const openRenameModal = (kb) => {
    setEditingKb(kb);
    setModalMode("edit");
  };

  const closeModal = () => {
    setModalMode(null);
    setEditingKb(null);
  };

  const handleSubmit = async ({ name, description }) => {
    if (modalMode === "edit" && editingKb) {
      await updateKnowledgeBase(editingKb.id, { name, description });
    } else {
      await createKnowledgeBase(name, description);
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <BookMarked className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Knowledge Bases
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Organize documents into collections you can chat against.
          </p>
        </div>

        <button
          onClick={openCreateModal}
          className="flex items-center space-x-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm rounded-xl transition-colors shadow-sm"
        >
          <Plus className="w-4 h-4" />
          <span>New Knowledge Base</span>
        </button>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400 dark:text-slate-500">
          <Spinner size="lg" />
          <span className="mt-3 text-xs">Loading knowledge bases...</span>
        </div>
      )}

      {!loading && error && (
        <div className="flex flex-col items-center justify-center py-20 text-center space-y-3">
          <div className="p-3 bg-red-50 border border-red-100 text-red-600 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400 rounded-2xl">
            <TriangleAlert className="w-8 h-8" />
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-300">{error}</p>
          <button
            onClick={fetchKnowledgeBases}
            className="px-4 py-2 text-xs font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 bg-blue-50 hover:bg-blue-100 dark:bg-blue-500/10 dark:hover:bg-blue-500/20 rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>
      )}

      {!loading && !error && knowledgeBases.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center space-y-3 max-w-sm mx-auto">
          <div className="p-3 bg-blue-50 border border-blue-100 text-blue-600 dark:bg-blue-600/10 dark:border-blue-500/20 dark:text-blue-400 rounded-2xl">
            <BookMarked className="w-8 h-8" />
          </div>
          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
            No knowledge bases yet
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Create one to group related documents and chat against just that collection.
          </p>
          <button
            onClick={openCreateModal}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs rounded-xl transition-colors shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New Knowledge Base</span>
          </button>
        </div>
      )}

      {!loading && !error && knowledgeBases.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {knowledgeBases.map((kb) => (
            <KnowledgeBaseCard
              key={kb.id}
              kb={kb}
              onRename={openRenameModal}
              onDelete={deleteKnowledgeBase}
            />
          ))}
        </div>
      )}

      {modalMode && (
        <KnowledgeBaseFormModal
          mode={modalMode}
          initialValues={editingKb}
          onSubmit={handleSubmit}
          onClose={closeModal}
        />
      )}
    </div>
  );
};
