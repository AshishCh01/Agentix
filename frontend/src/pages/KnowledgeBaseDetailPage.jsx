import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { ArrowLeft, BookMarked, MessageSquarePlus, Pencil, TriangleAlert } from "lucide-react";
import { knowledgeBaseApi } from "../api/knowledgeBaseApi";
import { useKnowledgeBase } from "../hooks/useKnowledgeBase";
import { useChat } from "../hooks/useChat";
import { useUI } from "../hooks/useUI";
import { usePageTitle } from "../hooks/usePageTitle";
import { Spinner } from "../components/Common/Loader";
import { Button, IconButton } from "../components/ui/Button";
import { StateMessage } from "../components/ui/StateMessage";
import { DocumentRow } from "../components/KnowledgeBase/DocumentRow";
import { KnowledgeBaseUpload } from "../components/KnowledgeBase/KnowledgeBaseUpload";
import { KnowledgeBaseFormModal } from "../components/KnowledgeBase/KnowledgeBaseFormModal";
import { getErrorMessage } from "../utils/helpers";

const POLL_INTERVAL_MS = 4000;

export default function KnowledgeBaseDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useUI();
  const { updateKnowledgeBase } = useKnowledgeBase();
  const { createNewSession } = useChat();

  const [kb, setKb] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showRename, setShowRename] = useState(false);
  const [startingChat, setStartingChat] = useState(false);
  const pollRef = useRef(null);

  usePageTitle(kb?.name || "Knowledge base");

  const loadAll = useCallback(async () => {
    const [kbData, docsData] = await Promise.all([
      knowledgeBaseApi.getKnowledgeBase(id),
      knowledgeBaseApi.getDocuments(id),
    ]);
    setKb(kbData);
    setDocuments(docsData);
  }, [id]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        await loadAll();
      } catch (err) {
        if (mounted) setError(getErrorMessage(err));
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [loadAll]);

  // Poll while any document is still processing; stops once nothing is pending.
  useEffect(() => {
    if (!documents.some((d) => d.status === "processing")) return;
    pollRef.current = setInterval(async () => {
      try {
        setDocuments(await knowledgeBaseApi.getDocuments(id));
      } catch {
        /* transient — next tick retries */
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [documents, id]);

  const handleUpload = async (file, onProgress) => {
    await knowledgeBaseApi.uploadDocument(id, file, onProgress);
    const docs = await knowledgeBaseApi.getDocuments(id);
    setDocuments(docs);
    setKb((prev) => (prev ? { ...prev, document_count: docs.length } : prev));
  };

  const handleDeleteDocument = async (documentId) => {
    await knowledgeBaseApi.deleteDocument(id, documentId);
    setDocuments((prev) => prev.filter((d) => d.id !== documentId));
    setKb((prev) => (prev ? { ...prev, document_count: Math.max(0, (prev.document_count || 1) - 1) } : prev));
    toast.success("Document deleted.");
  };

  const handleRename = async ({ name, description }) => {
    const updated = await updateKnowledgeBase(id, { name, description });
    setKb((prev) => ({ ...prev, ...updated }));
  };

  const handleStartChat = async () => {
    if (startingChat) return;
    setStartingChat(true);
    try {
      await createNewSession(`Chat: ${kb.name}`, id);
      navigate("/app");
    } catch (err) {
      toast.error(getErrorMessage(err) || "Couldn’t start a chat.");
      setStartingChat(false);
    }
  };

  if (loading) return <div className="flex justify-center py-32"><Spinner size="lg" /></div>;

  if (error) {
    return (
      <StateMessage
        className="py-32"
        icon={TriangleAlert}
        tone="danger"
        title="Couldn’t open this knowledge base"
        description={error}
        action={<Button as={Link} to="/app/knowledge-bases" variant="secondary">Back to knowledge bases</Button>}
      />
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-6 sm:px-6 sm:py-10">
      <Link to="/app/knowledge-bases" className="inline-flex items-center gap-1.5 text-sm text-ink-2 hover:text-ink">
        <ArrowLeft className="size-4" aria-hidden="true" />
        Knowledge bases
      </Link>

      <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <div className="mt-1 flex size-10 shrink-0 items-center justify-center rounded-lg border border-accent-line bg-accent-soft text-accent">
            <BookMarked className="size-5" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1">
              <h1 className="truncate font-display text-3xl leading-tight text-ink sm:text-4xl">{kb.name}</h1>
              <IconButton label="Edit name and description" size="sm" onClick={() => setShowRename(true)}>
                <Pencil className="size-3.5" />
              </IconButton>
            </div>
            <p className="mt-1 text-sm text-ink-2">{kb.description || "No description"}</p>
          </div>
        </div>
        <Button onClick={handleStartChat} loading={startingChat} className="sm:mt-1">
          {!startingChat && <MessageSquarePlus className="size-4" aria-hidden="true" />}
          Chat with this knowledge base
        </Button>
      </div>

      <section className="mt-10" aria-labelledby="upload-h">
        <h2 id="upload-h" className="label-mono mb-3">Add a document</h2>
        <KnowledgeBaseUpload onUpload={handleUpload} />
      </section>

      <section className="mt-10" aria-labelledby="docs-h">
        <h2 id="docs-h" className="label-mono mb-3">Documents · {documents.length}</h2>
        {documents.length === 0 ? (
          <p className="rounded-xl border border-dashed border-line-strong px-4 py-10 text-center text-sm text-ink-3">
            No documents yet. Upload one above to get started.
          </p>
        ) : (
          <ul className="space-y-2">
            {documents.map((doc) => (
              <DocumentRow key={doc.id} document={doc} onDelete={handleDeleteDocument} />
            ))}
          </ul>
        )}
      </section>

      {showRename && (
        <KnowledgeBaseFormModal mode="edit" initialValues={kb} onSubmit={handleRename} onClose={() => setShowRename(false)} />
      )}
    </div>
  );
}
