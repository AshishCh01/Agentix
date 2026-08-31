import { useState } from "react";
import { Plus } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { useKnowledgeBase } from "../../hooks/useKnowledgeBase";
import { Spinner } from "../Common/Loader";

export const NewChatButton = () => {
  const { createNewSession, isStreaming, defaultKnowledgeBaseId, setDefaultKnowledgeBaseId } = useChat();
  const { knowledgeBases } = useKnowledgeBase();
  const [loading, setLoading] = useState(false);

  const handleCreateNewChat = async () => {
    if (isStreaming || loading) return;
    setLoading(true);
    try {
      await createNewSession("New Chat Session");
    } catch (err) {
      console.error("Error creating new chat:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      {knowledgeBases.length > 0 && (
        <select
          value={defaultKnowledgeBaseId || ""}
          onChange={(e) => setDefaultKnowledgeBaseId(e.target.value || null)}
          title="Scope new chats to a knowledge base"
          className="w-full px-3 py-2 text-xs bg-slate-100 border border-slate-200 text-slate-700 dark:bg-slate-800 dark:border-slate-700/50 dark:text-slate-200 rounded-lg focus:outline-none focus:border-blue-500/80 focus:ring-1 focus:ring-blue-500/50 transition-all"
        >
          <option value="">No Knowledge Base</option>
          {knowledgeBases.map((kb) => (
            <option key={kb.id} value={kb.id}>
              {kb.name}
            </option>
          ))}
        </select>
      )}

      <button
        onClick={handleCreateNewChat}
        disabled={isStreaming || loading}
        className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition-colors shadow-sm"
      >
        {loading ? (
          <Spinner size="sm" className="text-white" />
        ) : (
          <>
            <Plus className="w-4 h-4" />
            <span>New Chat</span>
          </>
        )}
      </button>
    </div>
  );
};