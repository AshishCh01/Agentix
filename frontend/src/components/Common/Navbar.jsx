import { useAuth } from "../../hooks/useAuth";
import { useChat } from "../../hooks/useChat";
import { Bot, LogOut, User } from "lucide-react";
import { UploadButton } from "../Upload/UploadButton";

export const Navbar = ({ onToggleSidebar }) => {
  const { user, logout } = useAuth();
  const { sessions, activeSessionId } = useChat();

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur px-4 flex items-center justify-between sticky top-0 z-10">
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 lg:hidden"
        >
          <Bot className="w-6 h-6" />
        </button>
        <div>
          <h1 className="text-sm font-semibold text-slate-100 truncate max-w-[180px] sm:max-w-xs">
            {activeSession?.title || "Agentic RAG Assistant"}
          </h1>
          <p className="text-xs text-slate-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Multi-Agent Active
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <UploadButton />

        <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50 text-xs text-slate-300">
          <User className="w-3.5 h-3.5 text-blue-400" />
          <span className="truncate max-w-[150px]">{user?.email}</span>
        </div>

        <button
          onClick={logout}
          title="Sign Out"
          className="p-2 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded-lg transition-colors"
        >
          <LogOut className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
};