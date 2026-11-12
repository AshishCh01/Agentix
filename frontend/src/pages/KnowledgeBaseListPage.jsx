import { useState } from "react";
import { BookMarked, Plus, TriangleAlert } from "lucide-react";
import { useKnowledgeBase } from "../hooks/useKnowledgeBase";
import { usePageTitle } from "../hooks/usePageTitle";
import { Spinner } from "../components/Common/Loader";
import { Button } from "../components/ui/Button";
import { PageHeader } from "../components/ui/PageHeader";
import { StateMessage } from "../components/ui/StateMessage";
import { KnowledgeBaseCard } from "../components/KnowledgeBase/KnowledgeBaseCard";
import { KnowledgeBaseFormModal } from "../components/KnowledgeBase/KnowledgeBaseFormModal";

export default function KnowledgeBaseListPage() {
  usePageTitle("Knowledge bases");
  const { knowledgeBases, loading, error, fetchKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase } =
    useKnowledgeBase();
  const [modal, setModal] = useState(null); // null | "create" | "edit"
  const [editing, setEditing] = useState(null);

  const close = () => {
    setModal(null);
    setEditing(null);
  };
  const submit = ({ name, description }) =>
    modal === "edit" ? updateKnowledgeBase(editing.id, { name, description }) : createKnowledgeBase(name, description);

  const newButton = (
    <Button onClick={() => setModal("create")}>
      <Plus className="size-4" aria-hidden="true" />
      New knowledge base
    </Button>
  );

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-6 sm:py-10">
      <PageHeader
        title="Knowledge bases"
        description="Group documents together, then start a chat that searches only that group."
        actions={knowledgeBases.length > 0 && newButton}
      />

      <div className="mt-8">
        {loading && knowledgeBases.length === 0 ? (
          <div className="flex justify-center py-24"><Spinner size="lg" /></div>
        ) : error && knowledgeBases.length === 0 ? (
          <StateMessage
            icon={TriangleAlert}
            tone="danger"
            title="Couldn’t load your knowledge bases"
            description={typeof error === "string" ? error : "Check your connection and try again."}
            action={<Button variant="secondary" onClick={fetchKnowledgeBases}>Try again</Button>}
          />
        ) : knowledgeBases.length === 0 ? (
          <StateMessage
            icon={BookMarked}
            title="No knowledge bases yet"
            description="Create one, upload a few documents, and chat with just that set."
            action={newButton}
          />
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {knowledgeBases.map((kb) => (
              <li key={kb.id}>
                <KnowledgeBaseCard
                  kb={kb}
                  onRename={(k) => {
                    setEditing(k);
                    setModal("edit");
                  }}
                  onDelete={deleteKnowledgeBase}
                />
              </li>
            ))}
          </ul>
        )}
      </div>

      {modal && (
        <KnowledgeBaseFormModal
          mode={modal}
          initialValues={editing}
          onSubmit={submit}
          onClose={close}
        />
      )}
    </div>
  );
}
