/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useMemo, useRef, useState } from "react";
import { CheckCircle2, CircleAlert, Info, X } from "lucide-react";
import { Modal } from "../components/ui/Modal";
import { Button } from "../components/ui/Button";
import { cn } from "../utils/cn";

export const UIContext = createContext(null);

const ICONS = { success: CheckCircle2, error: CircleAlert, info: Info };
const TONES = {
  success: "text-accent",
  error: "text-danger",
  info: "text-ink-2",
};

/** App-wide toasts and an awaitable confirm dialog (replaces alert/confirm). */
export const UIProvider = ({ children }) => {
  const [toasts, setToasts] = useState([]);
  const [dialog, setDialog] = useState(null);
  const idRef = useRef(0);

  const dismiss = useCallback((id) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const push = useCallback(
    (type, message) => {
      const id = ++idRef.current;
      setToasts((t) => [...t.slice(-2), { id, type, message }]);
      setTimeout(() => dismiss(id), type === "error" ? 6500 : 4000);
    },
    [dismiss]
  );

  const confirm = useCallback(
    (options) => new Promise((resolve) => setDialog({ ...options, resolve })),
    []
  );

  const settle = (result) => {
    dialog?.resolve(result);
    setDialog(null);
  };

  const value = useMemo(
    () => ({
      confirm,
      toast: {
        success: (m) => push("success", m),
        error: (m) => push("error", m),
        info: (m) => push("info", m),
      },
    }),
    [confirm, push]
  );

  return (
    <UIContext.Provider value={value}>
      {children}

      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-0 top-3 z-[60] flex flex-col items-center gap-2 px-3 pt-safe sm:items-end sm:px-5"
      >
        {toasts.map((t) => {
          const Icon = ICONS[t.type];
          return (
            <div
              key={t.id}
              role={t.type === "error" ? "alert" : "status"}
              className="animate-pop pointer-events-auto flex w-full max-w-sm items-start gap-2.5 rounded-xl border border-line bg-surface px-3.5 py-3 text-sm text-ink shadow-lg"
            >
              <Icon className={cn("mt-0.5 size-4 shrink-0", TONES[t.type])} aria-hidden="true" />
              <p className="min-w-0 flex-1 leading-snug">{t.message}</p>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss"
                className="-m-1 rounded p-1 text-ink-3 hover:text-ink"
              >
                <X className="size-3.5" />
              </button>
            </div>
          );
        })}
      </div>

      {dialog && (
        <Modal title={dialog.title} description={dialog.description} onClose={() => settle(false)}>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => settle(false)}>
              Cancel
            </Button>
            <Button
              data-autofocus
              variant={dialog.tone === "danger" ? "danger" : "primary"}
              onClick={() => settle(true)}
            >
              {dialog.confirmLabel || "Confirm"}
            </Button>
          </div>
        </Modal>
      )}
    </UIContext.Provider>
  );
};
