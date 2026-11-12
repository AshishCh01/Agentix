import { useMemo } from "react";
import { useOutletContext } from "react-router-dom";
import { BookMarked, Menu } from "lucide-react";
import { useChat } from "../hooks/useChat";
import { useKnowledgeBase } from "../hooks/useKnowledgeBase";
import { usePageTitle } from "../hooks/usePageTitle";
import { IconButton } from "../components/ui/Button";
import { UploadButton } from "../components/Upload/UploadButton";
import { MessageList } from "../components/Chat/MessageList";
import { Composer } from "../components/Chat/Composer";

export default function ChatPage() {
  const { openMenu } = useOutletContext();
  const { sessions, activeSessionId } = useChat();
  const { knowledgeBases } = useKnowledgeBase();

  const session = sessions.find((s) => s.id === activeSessionId);
  const kb = useMemo(
    () => (session?.knowledge_base_id ? knowledgeBases.find((k) => k.id === session.knowledge_base_id) : null),
    [session, knowledgeBases]
  );
  usePageTitle(session?.title);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-1 border-b border-line px-2 pt-safe sm:px-4">
        <IconButton label="Open menu" onClick={openMenu} className="lg:hidden">
          <Menu className="size-5" />
        </IconButton>
        <div className="min-w-0 flex-1 px-1">
          <h1 className="truncate text-sm font-semibold text-ink">{session?.title || "New chat"}</h1>
          <p className="flex items-center gap-1 truncate text-xs text-ink-3">
            {kb ? (
              <>
                <BookMarked className="size-3 shrink-0 text-accent" aria-hidden="true" />
                <span className="truncate">{kb.name}</span>
              </>
            ) : (
              "No knowledge base"
            )}
          </p>
        </div>
        <UploadButton />
      </header>
      <MessageList />
      <Composer />
    </div>
  );
}
