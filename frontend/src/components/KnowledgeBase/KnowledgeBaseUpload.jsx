import { UploadPanel } from "../Upload/UploadPanel";

/** Upload dropzone scoped to one knowledge base; `onUpload(file, onProgress)` does the request. */
export const KnowledgeBaseUpload = ({ onUpload }) => (
  <div className="rounded-xl border border-line bg-surface p-4">
    <UploadPanel upload={onUpload} doneLabel="Done" onDone={undefined} />
  </div>
);
