import { CircleAlert, CheckCircle2, FileText, X } from "lucide-react";
import { Spinner } from "../Common/Loader";
import { IconButton } from "../ui/Button";
import { formatFileSize } from "../../utils/files";

export const FilePreview = ({ file, progress, isUploading, error, success, onClear }) => {
  if (!file) return null;
  return (
    <div className="space-y-3 rounded-xl border border-line bg-sunken p-3.5">
      <div className="flex items-center gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-line bg-surface text-accent">
          <FileText className="size-5" aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-ink" title={file.name}>{file.name}</p>
          <p className="text-xs text-ink-3">{formatFileSize(file.size)}</p>
        </div>
        <IconButton label="Remove file" size="sm" onClick={onClear} disabled={isUploading}>
          <X className="size-4" />
        </IconButton>
      </div>

      {isUploading && (
        <div className="space-y-1.5" role="status">
          <div className="flex items-center justify-between text-xs text-ink-2">
            <span className="inline-flex items-center gap-1.5"><Spinner size="sm" /> Uploading…</span>
            <span className="tabular-nums">{progress}%</span>
          </div>
          <div
            className="h-1.5 overflow-hidden rounded-full bg-line"
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="h-full bg-accent transition-[width] duration-200" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {success && (
        <p role="status" className="flex items-center gap-2 text-xs text-accent">
          <CheckCircle2 className="size-4 shrink-0" aria-hidden="true" />
          Uploaded. The file is being processed in the background.
        </p>
      )}

      {error && (
        <p role="alert" className="flex items-start gap-2 text-xs text-danger">
          <CircleAlert className="mt-px size-4 shrink-0" aria-hidden="true" />
          {error}
        </p>
      )}
    </div>
  );
};
