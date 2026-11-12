import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { useKnowledgeBase } from "../../hooks/useKnowledgeBase";
import { Spinner } from "../Common/Loader";
import ErrorBoundary from "../Common/ErrorBoundary";
import { EmptyState } from "./EmptyState";
import { MessageItem } from "./MessageItem";

const NEAR_BOTTOM_PX = 120;

export const MessageList = () => {
  const { messages, loadingMessages, isStreaming, activeNode, activeSessionId, sessions, sendMessage } = useChat();
  const { knowledgeBases } = useKnowledgeBase();
  const scrollRef = useRef(null);
  const stickRef = useRef(true);
  const [showJump, setShowJump] = useState(false);

  // Keep a stable callback so memoized items don't re-render when the provider value changes.
  const sendRef = useRef(sendMessage);
  useEffect(() => {
    sendRef.current = sendMessage;
  });
  const searchWeb = useCallback(
    (query) => sendRef.current(query, null, true, { skipUserMessageInTranscript: true }),
    []
  );

  const scrollToBottom = useCallback((behavior = "auto") => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  const onScroll = () => {
    const el = scrollRef.current;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    stickRef.current = near;
    setShowJump(!near);
  };

  // New chat selected → jump to the end of it.
  useLayoutEffect(() => {
    stickRef.current = true;
  }, [activeSessionId]);

  // Follow new content only while the reader is already at the bottom.
  useLayoutEffect(() => {
    const last = messages[messages.length - 1];
    const prev = messages[messages.length - 2];
    if (prev?.role === "user" && last?.isStreaming && !last.content) stickRef.current = true; // just sent
    if (stickRef.current) scrollToBottom();
  }, [messages, activeNode, loadingMessages, scrollToBottom]);

  const items = useMemo(() => {
    let latestAssistantId = null;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        latestAssistantId = messages[i].id;
        break;
      }
    }
    return messages.map((msg, idx) => {
      let previousQuery = "";
      if (msg.role === "assistant") {
        for (let i = idx - 1; i >= 0; i--) {
          if (messages[i].role === "user") {
            previousQuery = messages[i].content?.trim() || "";
            break;
          }
        }
      }
      return { msg, previousQuery, isLatestAssistant: msg.id === latestAssistantId };
    });
  }, [messages]);

  const scopeName = useMemo(() => {
    const kbId = sessions.find((s) => s.id === activeSessionId)?.knowledge_base_id;
    return kbId ? knowledgeBases.find((k) => k.id === kbId)?.name : null;
  }, [sessions, activeSessionId, knowledgeBases]);

  if (loadingMessages) {
    return (
      <div role="status" className="flex flex-1 flex-col items-center justify-center gap-2 text-ink-3">
        <Spinner size="lg" />
        <span className="text-xs">Loading messages…</span>
      </div>
    );
  }

  if (!messages.length) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto">
        <EmptyState scopeName={scopeName} disabled={isStreaming} onPick={(t) => sendMessage(t)} />
      </div>
    );
  }

  return (
    <div className="relative min-h-0 flex-1">
      <div ref={scrollRef} onScroll={onScroll} className="scroll-thin h-full overflow-y-auto overscroll-contain">
        <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-6 sm:px-6">
          {items.map(({ msg, previousQuery, isLatestAssistant }) => (
            <ErrorBoundary key={msg.id}>
              <MessageItem
                message={msg}
                previousQuery={previousQuery}
                isLatestAssistant={isLatestAssistant}
                busy={isStreaming}
                activeNode={msg.isStreaming ? activeNode : null}
                onSearchWeb={searchWeb}
              />
            </ErrorBoundary>
          ))}
        </div>
      </div>

      {showJump && (
        <button
          type="button"
          onClick={() => scrollToBottom("smooth")}
          aria-label="Jump to latest message"
          className="absolute bottom-3 left-1/2 flex size-9 -translate-x-1/2 items-center justify-center rounded-full border border-line-strong bg-surface text-ink-2 shadow-md transition hover:text-ink"
        >
          <ArrowDown className="size-4" />
        </button>
      )}
    </div>
  );
};
