import { useState } from "react";
import { Paperclip } from "lucide-react";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { UploadPanel } from "./UploadPanel";
import { useChat } from "../../hooks/useChat";
import { uploadApi } from "../../api/uploadApi";

/** "Add document" for the current chat. Creates a chat first if none is open. */
export const UploadButton = () => {
  const [open, setOpen] = useState(false);
  const { activeSessionId, createNewSession } = useChat();

  const upload = async (file, onProgress) => {
    let sessionId = activeSessionId;
    if (!sessionId) {
      const created = await createNewSession(`Doc: ${file.name.slice(0, 20)}`);
      sessionId = created.id;
    }
    await uploadApi.uploadDocument(sessionId, file, onProgress);
  };

  return (
    <>
      <Button variant="secondary" size="sm" onClick={() => setOpen(true)} aria-label="Add document">
        <Paperclip className="size-4" aria-hidden="true" />
        <span className="hidden sm:inline">Add document</span>
      </Button>
      {open && (
        <Modal
          title="Add a document"
          description="It will be available to this chat once processing finishes."
          onClose={() => setOpen(false)}
        >
          <UploadPanel upload={upload} onDone={() => setOpen(false)} />
        </Modal>
      )}
    </>
  );
};
