import { X, LogOut, User } from "lucide-react";
import { NewChatButton } from "./NewChatButton";
import { ChatList } from "./ChatList";
import { useAuth } from "../../hooks/useAuth";

export const Sidebar = ({ isOpen, onClose }) => {
  const { user, logout } = useAuth();

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-30 w-72 bg-slate-50 border-r border-slate-200 dark:bg-slate-950 dark:border-slate-800 flex flex-col transform transition-transform duration-200 ease-in-out lg:static lg:translate-x-0 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      {/* Sidebar Header */}
      <div className="h-16 px-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <img src="/Agentix_logo_mark.png" alt="Agentix" className="h-7 w-auto" />
          <span className="font-semibold text-sm text-slate-900 dark:text-slate-100 tracking-wide">
            Agentix
          </span>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 lg:hidden rounded-lg hover:bg-slate-200/60 dark:hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* New Chat Button Area */}
      <div className="p-3">
        <NewChatButton />
      </div>

      {/* Scrollable Session List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar py-2">
        <div className="px-4 pb-2 text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
          Chat History
        </div>
        <ChatList onSelectSession={onClose} />
      </div>

      {/* User Profile Footer */}
      <div className="p-3 border-t border-slate-200 dark:border-slate-800 bg-slate-100/60 dark:bg-slate-950/50 flex items-center justify-between">
        <div className="flex items-center space-x-2.5 truncate min-w-0 pr-2">
          <div className="w-8 h-8 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center justify-center flex-shrink-0 text-blue-600 dark:text-blue-400">
            <User className="w-4 h-4" />
          </div>
          <div className="truncate">
            <p className="text-xs font-medium text-slate-700 dark:text-slate-200 truncate">
              {user?.email}
            </p>
            <p className="text-[10px] text-slate-400 dark:text-slate-500">Authenticated</p>
          </div>
        </div>

        <button
          onClick={logout}
          title="Sign Out"
          className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-slate-200/60 dark:text-slate-400 dark:hover:text-red-400 dark:hover:bg-slate-800 rounded-lg transition-colors flex-shrink-0"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;