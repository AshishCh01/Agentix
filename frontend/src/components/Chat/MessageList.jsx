import { useEffect, useMemo, useRef } from "react";
import { Bot, Sparkles } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { MessageItem } from "./MessageItem";
import { NodeIndicator } from "./NodeIndicator";
import { Spinner } from "../Common/Loader";
import ErrorBoundary from "../Common/ErrorBoundary"; // 1. Added import

export const MessageList = () => {
  const { messages, loadingMessages, isStreaming, activeNode } = useChat();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeNode]);

  // Recomputed only when `messages` itself changes -- not on every
  // activeNode/isStreaming update this component also re-renders for --
  // and MessageItem is memoized, so a token appended to the last message
  // re-renders just that one item instead of the whole list.
  const messageItems = useMemo(
    () =>
      messages.map((msg) => (
        <ErrorBoundary key={msg.id}>
          <MessageItem message={msg} />
        </ErrorBoundary>
      )),
    [messages]
  );

  if (loadingMessages) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-2">
        <Spinner size="lg" />
        <span className="text-xs">Loading message history...</span>
      </div>
    );
  }

  if (!messages || messages.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4 max-w-md mx-auto">
        <div className="p-3 bg-blue-600/10 border border-blue-500/20 text-blue-400 rounded-2xl">
          <Bot className="w-10 h-10" />
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-slate-200">
            How can I help you today?
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Ask questions about your uploaded PDF documents, requested general information, or initiate a live web search.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-4 max-w-4xl mx-auto">
      {messageItems}

      {/* Dynamic Agent Streaming Indicator */}
      {isStreaming && (
        <div className="flex space-x-3 max-w-4xl mx-auto">
          <div className="w-8 h-8 rounded-xl bg-blue-600/10 border border-blue-500/20 text-blue-400 flex items-center justify-center flex-shrink-0 mt-0.5">
            <Sparkles className="w-4 h-4 animate-spin" />
          </div>
          <div className="flex-1">
            <NodeIndicator node={activeNode} />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
};