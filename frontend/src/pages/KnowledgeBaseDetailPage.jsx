import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { ArrowLeft, BookMarked, MessageSquarePlus, Pencil, TriangleAlert } from "lucide-react";
import { knowledgeBaseApi } from "../api/knowledgeBaseApi";
import { useKnowledgeBase } from "../hooks/useKnowledgeBase";
import { useChat } from "../hooks/useChat";
import { Spinner } from "../components/Common/Loader";
import { DocumentRow } from "../components/KnowledgeBase/DocumentRow";
import { KnowledgeBaseUpload } from "../components/KnowledgeBase/KnowledgeBaseUpload";
import { KnowledgeBaseFormModal } from "../components/KnowledgeBase/KnowledgeBaseFormModal";
import { getErrorMessage } from "../utils/helpers";

const POLL_INTERVAL_MS = 4000;

export const KnowledgeBaseDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { updateKnowledgeBase } = useKnowledgeBase();
  const { createNewSession } = useChat();

  const [kb, setKb] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showRename, setShowRename] = useState(false);
  const [isStartingChat, setIsStartingChat] = useState(false);

  const pollRef = useRef(null);

  const loadAll = useCallback(async () => {
    const [kbData, docsData] = await Promise.all([
      knowledgeBaseApi.getKnowledgeBase(id),
      knowledgeBaseApi.getDocuments(id),
    ]);
    setKb(kbData);
    setDocuments(docsData);
    return docsData;
  }, [id]);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        await loadAll();
      } catch (err) {
        if (isMounted) setError(getErrorMessage(err));
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    load();

    return () => {
      isMounted = false;
    };
  }, [loadAll]);

  // Poll while any document is still processing, so status/progress badges
  // update without a manual refresh -- stops itself once nothing is pending.
  useEffect(() => {
    const hasProcessing = documents.some((d) => d.status === "processing");
    if (!hasProcessing) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const docsData = await knowledgeBaseApi.getDocuments(id);
        setDocuments(docsData);
      } catch {
        // Transient poll failures are ignored -- the next tick retries.
      }
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documents, id]);

  const handleUpload = async (file, onProgress) => {
    await knowledgeBaseApi.uploadDocument(id, file, onProgress);
    const docsData = await knowledgeBaseApi.getDocuments(id);
    setDocuments(docsData);
    setKb((prev) => (prev ? { ...prev, document_count: docsData.length } : prev));
  };

  const handleDeleteDocument = async (documentId) => {
    await knowledgeBaseApi.deleteDocument(id, documentId);
    setDocuments((prev) => prev.filter((d) => d.id !== documentId));
    setKb((prev) => (prev ? { ...prev, document_count: prev.document_count - 1 } : prev));
  };

  const handleRename = async ({ name, description }) => {
    const updated = await updateKnowledgeBase(id, { name, description });
    setKb((prev) => ({ ...prev, ...updated }));
  };

  const handleStartChat = async () => {
    if (isStartingChat) return;
    setIsStartingChat(true);
    try {
      await createNewSession(`Chat: ${kb.name}`, id);
      navigate("/");
    } catch (err) {
      console.error("Failed to start chat scoped to knowledge base:", err);
      setIsStartingChat(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-slate-400 dark:text-slate-500">
        <Spinner size="lg" />
        <span className="mt-3 text-xs">Loading knowledge base...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center space-y-3">
        <div className="p-3 bg-red-50 border border-red-100 text-red-600 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400 rounded-2xl">
          <TriangleAlert className="w-8 h-8" />
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-300">{error}</p>
        <Link
          to="/knowledge-bases"
          className="px-4 py-2 text-xs font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 bg-blue-50 hover:bg-blue-100 dark:bg-blue-500/10 dark:hover:bg-blue-500/20 rounded-lg transition-colors"
        >
          Back to Knowledge Bases
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div>
        <Link
          to="/knowledge-bases"
          className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 mb-3 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Knowledge Bases
        </Link>

        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start space-x-3 min-w-0">
            <div className="p-2.5 bg-blue-50 border border-blue-100 text-blue-600 dark:bg-blue-600/10 dark:border-blue-500/20 dark:text-blue-400 rounded-xl flex-shrink-0">
              <BookMarked className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100 truncate">
                  {kb.name}
                </h1>
                <button
                  onClick={() => setShowRename(true)}
                  title="Rename"
                  className="p-1 text-slate-400 hover:text-blue-600 hover:bg-slate-100 dark:text-slate-500 dark:hover:text-blue-400 dark:hover:bg-slate-800 rounded-lg transition-colors flex-shrink-0"
                >
                  <Pencil className="w-3.5 h-3.5" />
                </button>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {kb.description || "No description"}
              </p>
            </div>
          </div>

          <button
            onClick={handleStartChat}
            disabled={isStartingChat}
            className="flex items-center space-x-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition-colors shadow-sm flex-shrink-0"
          >
            {isStartingChat ? <Spinner size="sm" className="text-white" /> : <MessageSquarePlus className="w-4 h-4" />}
            <span>Chat with this KB</span>
          </button>
        </div>
      </div>

      <div className="bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-slate-800 rounded-2xl p-4">
        <h2 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
          Upload Document
        </h2>
        <KnowledgeBaseUpload onUpload={handleUpload} />
      </div>

      <div>
        <h2 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
          Documents ({documents.length})
        </h2>

        {documents.length === 0 ? (
          <div className="text-center py-10 px-4 text-slate-400 dark:text-slate-500 text-xs border border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
            No documents yet. Upload one above to get started.
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <DocumentRow key={doc.id} document={doc} onDelete={handleDeleteDocument} />
            ))}
          </div>
        )}
      </div>

      {showRename && (
        <KnowledgeBaseFormModal
          mode="edit"
          initialValues={kb}
          onSubmit={handleRename}
          onClose={() => setShowRename(false)}
        />
      )}
    </div>
  );
};
