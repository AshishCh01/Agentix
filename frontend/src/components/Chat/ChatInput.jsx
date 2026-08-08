import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles } from "lucide-react";
import { useChat } from "../../hooks/useChat";

export const ChatInput = () => {
  const [prompt, setPrompt] = useState("");
  const { sendMessage, isStreaming } = useChat();
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        160
      )}px`;
    }
  }, [prompt]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!prompt.trim() || isStreaming) return;
    const textToSend = prompt;
    setPrompt("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    sendMessage(textToSend);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 bg-slate-900/90 border-t border-slate-800 backdrop-blur sticky bottom-0 z-10">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative">
        <div className="relative flex items-center bg-slate-950 border border-slate-800 rounded-2xl focus-within:border-blue-500/80 focus-within:ring-1 focus-within:ring-blue-500/50 transition-all shadow-xl">
          <textarea
            ref={textareaRef}
            rows={1}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question or upload a document..."
            disabled={isStreaming}
            className="w-full py-3.5 pl-4 pr-12 bg-transparent text-sm text-slate-100 placeholder-slate-500 resize-none focus:outline-none max-h-40 custom-scrollbar"
          />

          <button
            type="submit"
            disabled={!prompt.trim() || isStreaming}
            className="absolute right-2.5 p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-xl transition-all flex items-center justify-center flex-shrink-0"
          >
            {isStreaming ? (
              <Sparkles className="w-4 h-4 animate-spin text-blue-400" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-500 px-2 mt-2">
          <span>Press Enter to send, Shift + Enter for new line</span>
          <span>Powered by Agentic RAG Pipeline</span>
        </div>
      </form>
    </div>
  );
};