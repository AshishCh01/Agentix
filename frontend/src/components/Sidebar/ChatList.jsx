import { MessageSquare, Trash2 } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { Spinner } from "../Common/Loader";
import { truncateText } from "../../utils/helpers";

export const ChatList = ({ onSelectSession }) => {
  const {
    sessions,
    activeSessionId,
    setActiveSessionId,
    deleteSession,
    loadingSessions,
    isStreaming,
  } = useChat();

  const handleSelect = (sessionId) => {
    if (isStreaming) return;
    setActiveSessionId(sessionId);
    if (onSelectSession) onSelectSession();
  };

  const handleDelete = async (e, sessionId) => {
    e.stopPropagation();
    if (isStreaming) return;
    if (window.confirm("Are you sure you want to delete this chat session?")) {
      try {
        await deleteSession(sessionId);
      } catch (err) {
        console.error("Failed to delete session:", err);
      }
    }
  };

  if (loadingSessions) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-slate-500">
        <Spinner size="sm" />
        <span className="mt-2 text-xs">Loading sessions...</span>
      </div>
    );
  }

  if (!sessions || sessions.length === 0) {
    return (
      <div className="text-center py-8 px-4 text-slate-500 text-xs">
        No active chat sessions. Click "New Chat" above to start.
      </div>
    );
  }

  return (
    <div className="space-y-1 px-2">
      {sessions.map((session) => {
        const isActive = session.id === activeSessionId;
        return (
          <div
            key={session.id}
            onClick={() => handleSelect(session.id)}
            className={`group flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer transition-colors text-xs font-medium ${
              isActive
                ? "bg-slate-800 text-slate-100"
                : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
            }`}
          >
            <div className="flex items-center space-x-2.5 truncate min-w-0 pr-2">
              <MessageSquare
                className={`w-4 h-4 flex-shrink-0 ${
                  isActive ? "text-blue-400" : "text-slate-500"
                }`}
              />
              <span className="truncate">{truncateText(session.title, 26)}</span>
            </div>

            <button
              onClick={(e) => handleDelete(e, session.id)}
              title="Delete session"
              className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 hover:bg-slate-700/50 rounded transition-all flex-shrink-0"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
};