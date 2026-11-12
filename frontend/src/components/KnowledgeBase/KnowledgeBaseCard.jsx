import { useState } from "react";
import { Link } from "react-router-dom";
import { BookMarked, FileText, Pencil, Trash2 } from "lucide-react";
import { IconButton } from "../ui/Button";
import { useUI } from "../../hooks/useUI";
import { formatDate, getErrorMessage, truncateText } from "../../utils/helpers";

export const KnowledgeBaseCard = ({ kb, onRename, onDelete }) => {
  const { confirm, toast } = useUI();
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    if (deleting) return;
    const ok = await confirm({
      title: `Delete “${kb.name}”?`,
      description: "This also removes every document in it. This can’t be undone.",
      confirmLabel: "Delete",
      tone: "danger",
    });
    if (!ok) return;
    setDeleting(true);
    try {
      await onDelete(kb.id);
      toast.success("Knowledge base deleted.");
    } catch (err) {
      toast.error(getErrorMessage(err) || "Couldn’t delete the knowledge base.");
      setDeleting(false);
    }
  };

  const count = kb.document_count ?? 0;

  return (
    <article className="group relative flex flex-col rounded-xl border border-line bg-surface p-4 transition hover:border-accent-line focus-within:border-accent-line">
      <div className="flex items-start justify-between gap-2">
        <div className="flex size-9 items-center justify-center rounded-lg border border-accent-line bg-accent-soft text-accent">
          <BookMarked className="size-[18px]" aria-hidden="true" />
        </div>
        {/* Sit above the stretched link; always visible so touch users can reach them */}
        <div className="relative z-10 -mr-1.5 -mt-1 flex items-center">
          <IconButton label={`Rename ${kb.name}`} size="sm" onClick={() => onRename(kb)}>
            <Pencil className="size-3.5" />
          </IconButton>
          <IconButton
            label={`Delete ${kb.name}`}
            size="sm"
            disabled={deleting}
            onClick={handleDelete}
            className="hover:text-danger"
          >
            <Trash2 className="size-3.5" />
          </IconButton>
        </div>
      </div>

      <h3 className="mt-3 truncate text-[15px] font-semibold text-ink">
        <Link to={`/app/knowledge-bases/${kb.id}`} className="after:absolute after:inset-0 after:rounded-xl focus-visible:outline-none">
          {kb.name}
        </Link>
      </h3>
      <p className="mt-1 line-clamp-2 min-h-10 text-[13px] leading-5 text-ink-2">
        {kb.description ? truncateText(kb.description, 110) : "No description"}
      </p>

      <div className="mt-4 flex items-center justify-between gap-2 border-t border-line pt-3 text-xs text-ink-3">
        <span className="inline-flex items-center gap-1.5">
          <FileText className="size-3.5" aria-hidden="true" />
          {count} {count === 1 ? "document" : "documents"}
        </span>
        {kb.updated_at && <span className="truncate">Updated {formatDate(kb.updated_at)}</span>}
      </div>
    </article>
  );
};
