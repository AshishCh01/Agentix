import { useCallback, useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Menu } from "lucide-react";
import { Sidebar } from "../components/Sidebar/Sidebar";
import { IconButton } from "../components/ui/Button";
import { Logo } from "../components/ui/Logo";
import { cn } from "../utils/cn";

/** Signed-in shell: persistent sidebar on desktop, slide-over drawer on phones/tablets. */
export default function AppLayout() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const close = useCallback(() => setOpen(false), []);
  const openMenu = useCallback(() => setOpen(true), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && close();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      {open && <div className="fixed inset-0 z-30 bg-black/45 lg:hidden" onClick={close} aria-hidden="true" />}

      <aside
        id="app-sidebar"
        aria-label="Sidebar"
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-[18.5rem] max-w-[86vw] border-r border-line bg-surface",
          "transition-[transform,visibility] duration-200 ease-out lg:static lg:visible lg:w-72 lg:max-w-none lg:translate-x-0",
          open ? "translate-x-0" : "invisible -translate-x-full"
        )}
      >
        <Sidebar onNavigate={close} onClose={close} />
      </aside>

      <main id="main" className="flex min-w-0 flex-1 flex-col">
        {/* The chat page has its own header; other pages get a slim mobile bar. */}
        {pathname !== "/app" && (
          <div className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-2 pt-safe lg:hidden">
            <IconButton label="Open menu" onClick={openMenu} aria-expanded={open} aria-controls="app-sidebar">
              <Menu className="size-5" />
            </IconButton>
            <Logo to="/app" />
          </div>
        )}
        <div className={cn("scroll-thin flex min-h-0 flex-1 flex-col", pathname === "/app" ? "overflow-hidden" : "overflow-y-auto")}>
          <Outlet context={{ openMenu }} />
        </div>
      </main>
    </div>
  );
}
