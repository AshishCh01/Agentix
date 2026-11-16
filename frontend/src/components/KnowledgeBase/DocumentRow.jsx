import { useState } from "react";
import { CheckCircle2, FileText, Loader2, Trash2, TriangleAlert } from "lucide-react";
import { IconButton } from "../ui/Button";
import { useUI } from "../../hooks/useUI";
import { cn } from "../../utils/cn";
import { formatFileSize } from "../../utils/files";
import { formatDate, getErrorMessage } from "../../utils/helpers";

const STATUS = {
  processing: { icon: Loader2, spin: true, label: "Processing", cls: "border-warn/30 bg-warn-soft text-warn" },
  ready: { icon: CheckCircle2, label: "Ready", cls: "border-accent-line bg-accent-soft text-accent" },
  failed: { icon: TriangleAlert, label: "Failed", cls: "border-danger/30 bg-danger-soft text-danger" },
};

export const DocumentRow = ({ document, onDelete }) => {
  const { confirm, toast } = useUI();
  const [deleting, setDeleting] = useState(false);
  const status = STATUS[document.status] || STATUS.processing;
  const StatusIcon = status.icon;

  const handleDelete = async () => {
    if (deleting) return;
    const ok = await confirm({
      title: `Delete “${document.filename}”?`,
      description: "Chats scoped to this knowledge base will no longer be able to use it.",
      confirmLabel: "Delete",
      tone: "danger",
    });
    if (!ok) return;
    setDeleting(true);
    try {
      await onDelete(document.id);
    } catch (err) {
      toast.error(getErrorMessage(err) || "Couldn’t delete the document.");
      setDeleting(false);
    }
  };

  return (
    <li className="flex items-center gap-3 rounded-xl border border-line bg-surface px-3.5 py-3">
      <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-sunken text-ink-2">
        <FileText className="size-[18px]" aria-hidden="true" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-ink" title={document.filename}>{document.filename}</p>
        <p className="truncate text-xs text-ink-3">
          {formatFileSize(document.file_size)} · Uploaded {formatDate(document.created_at)}
        </p>
        {document.status === "failed" && (
          <p className="mt-0.5 line-clamp-2 text-xs text-danger">{document.error_message || "Ingestion failed"}</p>
        )}
      </div>
      <span className={cn("inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium", status.cls)}>
        <StatusIcon className={cn("size-3.5", status.spin && "animate-spin")} aria-hidden="true" />
        <span className="hidden sm:inline">{status.label}</span>
        <span className="sr-only sm:hidden">{status.label}</span>
      </span>
      <IconButton label={`Delete ${document.filename}`} size="sm" disabled={deleting} onClick={handleDelete} className="hover:text-danger">
        <Trash2 className="size-4" />
      </IconButton>
    </li>
  );
};
