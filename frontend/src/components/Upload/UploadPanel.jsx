import { useRef, useState } from "react";
import { FileUp } from "lucide-react";
import { Button } from "../ui/Button";
import { FilePreview } from "./FilePreview";
import { useUI } from "../../hooks/useUI";
import { cn } from "../../utils/cn";
import { getErrorMessage } from "../../utils/helpers";
import { ACCEPT_ATTR, MAX_UPLOAD_MB, SUPPORTED_LABEL, validateFile } from "../../utils/files";

/**
 * Drop zone + preview + upload button, shared by the chat "Add document" dialog
 * and the knowledge-base page. `upload(file, onProgress)` does the actual request.
 */
export const UploadPanel = ({ upload, onDone, doneLabel = "Done" }) => {
  const { toast } = useUI();
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const reset = () => {
    setBusy(false);
    setProgress(0);
    setError(null);
    setSuccess(false);
  };

  const accept = (candidate) => {
    if (!candidate) return;
    const problem = validateFile(candidate);
    if (problem) {
      toast.error(problem);
      return;
    }
    setFile(candidate);
    reset();
  };

  const clear = () => {
    setFile(null);
    reset();
    if (inputRef.current) inputRef.current.value = "";
  };

  const submit = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      await upload(file, setProgress);
      setSuccess(true);
    } catch (err) {
      setError(getErrorMessage(err) || "Document upload failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => accept(e.target.files?.[0])}
      />

      {!file ? (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            accept(e.dataTransfer.files?.[0]);
          }}
          className={cn(
            "flex w-full flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-9 text-center transition",
            dragging ? "border-accent bg-accent-soft" : "border-line-strong bg-sunken hover:border-accent"
          )}
        >
          <FileUp className="size-6 text-accent" aria-hidden="true" />
          <span className="text-sm font-medium text-ink">
            <span className="sm:hidden">Choose a file</span>
            <span className="hidden sm:inline">Drop a file here, or click to browse</span>
          </span>
          <span className="max-w-xs text-xs text-ink-3">
            {SUPPORTED_LABEL}. Up to {MAX_UPLOAD_MB} MB.
          </span>
        </button>
      ) : (
        <FilePreview
          file={file}
          progress={progress}
          isUploading={busy}
          error={error}
          success={success}
          onClear={clear}
        />
      )}

      <div className="flex justify-end gap-2">
        {success ? (
          <>
            <Button variant="secondary" onClick={clear}>Add another</Button>
            {onDone && <Button onClick={onDone}>{doneLabel}</Button>}
          </>
        ) : (
          <Button onClick={submit} disabled={!file} loading={busy}>
            {busy ? "Uploading…" : "Upload"}
          </Button>
        )}
      </div>
    </div>
  );
};
