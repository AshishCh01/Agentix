import { useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { BookMarked, LogOut, MessageSquare, Plus, Trash2, X } from "lucide-react";
import { useAuth } from "../../hooks/useAuth";
import { useChat } from "../../hooks/useChat";
import { useKnowledgeBase } from "../../hooks/useKnowledgeBase";
import { useUI } from "../../hooks/useUI";
import { Button, IconButton } from "../ui/Button";
import { Logo } from "../ui/Logo";
import { Spinner } from "../Common/Loader";
import { ThemeToggle } from "../Common/ThemeToggle";
import { cn } from "../../utils/cn";
import { truncateText } from "../../utils/helpers";

const navItem = ({ isActive }) =>
  cn(
    "flex h-10 items-center gap-2.5 rounded-lg px-3 text-sm font-medium transition",
    isActive ? "bg-sunken text-ink" : "text-ink-2 hover:bg-sunken hover:text-ink"
  );

export const Sidebar = ({ onNavigate, onClose }) => {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { user, logout } = useAuth();
  const { knowledgeBases } = useKnowledgeBase();
  const { confirm, toast } = useUI();
  const {
    sessions, activeSessionId, setActiveSessionId, createNewSession, deleteSession,
    loadingSessions, isStreaming, defaultKnowledgeBaseId, setDefaultKnowledgeBaseId,
  } = useChat();
  const [creating, setCreating] = useState(false);

  const go = (to) => {
    navigate(to);
    onNavigate?.();
  };

  const newChat = async () => {
    if (isStreaming || creating) return;
    setCreating(true);
    try {
      await createNewSession("New chat");
      go("/app");
    } catch {
      toast.error("Couldn’t start a new chat. Please try again.");
    } finally {
      setCreating(false);
    }
  };

  const removeSession = async (session) => {
    const ok = await confirm({
      title: "Delete this chat?",
      description: `“${truncateText(session.title, 40)}” and its messages will be removed.`,
      confirmLabel: "Delete",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await deleteSession(session.id);
    } catch {
      toast.error("Couldn’t delete the chat. Please try again.");
    }
  };

  const signOut = async () => {
    try {
      await logout();
    } catch {
      /* local state is already cleared by logout() */
    }
    navigate("/", { replace: true });
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-14 shrink-0 items-center justify-between pl-4 pr-2 pt-safe">
        <Logo to="/app" />
        <IconButton label="Close menu" onClick={onClose} className="lg:hidden">
          <X className="size-5" />
        </IconButton>
      </div>

      <div className="space-y-2 px-3 pb-3">
        <Button onClick={newChat} loading={creating} disabled={isStreaming} className="w-full">
          {!creating && <Plus className="size-4" aria-hidden="true" />}
          New chat
        </Button>
        {knowledgeBases.length > 0 && (
          <div>
            <label htmlFor="kb-scope" className="label-mono mb-1 block">New chats search in</label>
            <select
              id="kb-scope"
              value={defaultKnowledgeBaseId || ""}
              onChange={(e) => setDefaultKnowledgeBaseId(e.target.value || null)}
              className="h-10 w-full rounded-lg border border-line-strong bg-surface px-2.5 text-ink focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25"
            >
              <option value="">No knowledge base</option>
              {knowledgeBases.map((kb) => (
                <option key={kb.id} value={kb.id}>{kb.name}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <nav aria-label="Workspace" className="px-3">
        <NavLink to="/app/knowledge-bases" onClick={onNavigate} className={navItem}>
          <BookMarked className="size-4" aria-hidden="true" />
          Knowledge bases
        </NavLink>
      </nav>

      <div className="label-mono px-6 pb-1.5 pt-5">Chats</div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3">
        {loadingSessions && sessions.length === 0 ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : sessions.length === 0 ? (
          <p className="px-3 py-2 text-sm text-ink-3">No chats yet. Start one above.</p>
        ) : (
          <ul className="space-y-0.5 pb-3">
            {sessions.map((s) => {
              const active = pathname === "/app" && s.id === activeSessionId;
              return (
                <li key={s.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveSessionId(s.id);
                      go("/app");
                    }}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "flex h-10 w-full items-center gap-2.5 rounded-lg pl-3 pr-10 text-left text-sm transition",
                      active ? "bg-sunken font-medium text-ink" : "text-ink-2 hover:bg-sunken hover:text-ink"
                    )}
                  >
                    <MessageSquare className={cn("size-4 shrink-0", active && "text-accent")} aria-hidden="true" />
                    <span className="truncate">{s.title || "Untitled chat"}</span>
                  </button>
                  {/* Visible on touch; revealed on hover/focus for pointer devices */}
                  <IconButton
                    label={`Delete chat: ${s.title || "Untitled"}`}
                    size="sm"
                    onClick={() => removeSession(s)}
                    className="absolute right-1 top-1 hover:text-danger [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100 focus-visible:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </IconButton>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1 border-t border-line p-3 pb-safe">
        <div className="min-w-0 flex-1 px-1">
          <p className="truncate text-[13px] font-medium text-ink" title={user?.email}>{user?.email}</p>
        </div>
        <ThemeToggle />
        <IconButton label="Sign out" onClick={signOut}>
          <LogOut className="size-[18px]" />
        </IconButton>
      </div>
    </div>
  );
};
