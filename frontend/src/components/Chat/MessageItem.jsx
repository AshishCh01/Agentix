import { lazy, memo, Suspense, useDeferredValue, useState } from "react";
import { Check, CircleAlert, Copy, Globe } from "lucide-react";
import { LogoMark } from "../ui/Logo";
import { Button } from "../ui/Button";
import { SourceList } from "./SourceList";
import { VerificationBadge } from "./VerificationBadge";
import { AgentActivity } from "./AgentActivity";
import { isNotCoveredReply } from "../../utils/chatHelpers";

// Heavy (markdown + KaTeX + highlighting): loaded on first render of a message.
const Markdown = lazy(() => import("./Markdown"));

const AssistantBody = ({ message, activeNode }) => {
  // Defer re-parsing markdown while tokens arrive so typing stays smooth.
  const content = useDeferredValue(message.content || "");
  const isError = !message.isStreaming && /^Error:/.test(message.content || "");

  if (isError) {
    return (
      <div role="alert" className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger-soft p-3 text-sm text-danger">
        <CircleAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
        <p className="min-w-0 break-words">{message.content.replace(/^Error:\s*/, "") || "Something went wrong."}</p>
      </div>
    );
  }

  return (
    <>
      {content && (
        <Suspense fallback={<p className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink">{content}</p>}>
          <Markdown content={content} />
        </Suspense>
      )}
      {message.isStreaming && (
        <div className={content ? "mt-3" : ""}>
          <AgentActivity node={activeNode} />
        </div>
      )}
      {message.stopped && !content && <p className="text-sm italic text-ink-3">Stopped before an answer was written.</p>}
      {message.stopped && content && <p className="mt-2 text-xs italic text-ink-3">Stopped</p>}
    </>
  );
};

// Props are primitives/stable refs so only the streaming message re-renders per token.
export const MessageItem = memo(function MessageItem({
  message,
  isLatestAssistant = false,
  activeNode = null,
  busy = false,
  previousQuery = "",
  onSearchWeb,
}) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [searched, setSearched] = useState(false);

  const notCovered = !isUser && !message.isStreaming && isNotCoveredReply(message.content);
  const canSearchWeb = notCovered && isLatestAssistant && Boolean(previousQuery);
  const searchLocked = busy || searched || Boolean(message.webSearchInitiated);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard unavailable */
    }
  };

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[88%] space-y-2 rounded-2xl rounded-br-md border border-accent-line bg-accent-soft px-4 py-2.5 sm:max-w-[75%]">
          {message.image_data && (
            <img
              src={message.image_data}
              alt="Attached by you"
              className="max-h-56 max-w-full rounded-lg border border-line object-cover"
            />
          )}
          {message.content && (
            <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed text-ink">{message.content}</p>
          )}
        </div>
      </div>
    );
  }

  const showMeta = !message.isStreaming && !notCovered;

  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border border-line bg-surface">
        <LogoMark className="h-4" />
      </div>

      <div className="min-w-0 flex-1 space-y-3">
        <AssistantBody message={message} activeNode={activeNode} />

        {canSearchWeb && (
          <div className="flex flex-wrap items-center gap-3 border-t border-line pt-3">
            <Button
              variant="secondary"
              size="sm"
              disabled={searchLocked}
              onClick={() => {
                setSearched(true);
                onSearchWeb?.(previousQuery);
              }}
            >
              <Globe className="size-3.5" aria-hidden="true" />
              {searchLocked && !busy ? "Searching the web…" : "Search the web instead"}
            </Button>
            <span className="text-xs text-ink-3">Not found in your documents</span>
          </div>
        )}

        {!message.isStreaming && message.sources?.length > 0 && <SourceList sources={message.sources} />}

        {showMeta && (
          <div className="flex flex-wrap items-center gap-2">
            <VerificationBadge message={message} />
            {message.content && !/^Error:/.test(message.content) && (
              <button
                type="button"
                onClick={copy}
                className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-ink-3 hover:bg-sunken hover:text-ink"
              >
                {copied ? <Check className="size-3.5 text-accent" /> : <Copy className="size-3.5" />}
                {copied ? "Copied" : "Copy"}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
