import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Globe, ImagePlus, Square, X } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { useUI } from "../../hooks/useUI";
import { cn } from "../../utils/cn";
import { MAX_UPLOAD_MB } from "../../utils/files";
import { MAX_UPLOAD_BYTES } from "../../utils/constants";

const MAX_HEIGHT_PX = 200;

export const Composer = () => {
  const { sendMessage, isStreaming, cancelStream } = useChat();
  const { toast } = useUI();
  const [text, setText] = useState("");
  const [image, setImage] = useState(null); // data URL
  const [webSearch, setWebSearch] = useState(false);
  const taRef = useRef(null);
  const fileRef = useRef(null);

  // On touch keyboards Enter should add a new line; sending uses the button.
  const enterSends = useMemo(() => !window.matchMedia?.("(pointer: coarse)").matches, []);
  const canSend = (text.trim() || image) && !isStreaming;

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [text]);

  const attach = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return toast.error("Only image files can be attached to a message.");
    if (file.size > MAX_UPLOAD_BYTES) return toast.error(`That image is too large. The limit is ${MAX_UPLOAD_MB} MB.`);
    const reader = new FileReader();
    reader.onload = () => setImage(reader.result);
    reader.onerror = () => toast.error("Couldn’t read that image.");
    reader.readAsDataURL(file);
  };

  const submit = async () => {
    if (!canSend) return;
    const payload = { text: text.trim(), image };
    setText("");
    setImage(null);
    try {
      await sendMessage(payload.text, payload.image, webSearch);
    } catch {
      toast.error("Your message couldn’t be sent. Please try again.");
      setText(payload.text);
      setImage(payload.image);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && enterSends && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const onPaste = (e) => {
    const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
    if (file) {
      e.preventDefault();
      attach(file);
    }
  };

  return (
    <div className="shrink-0 px-3 pb-safe pt-2 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        <div className="rounded-2xl border border-line-strong bg-surface shadow-sm transition focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20">
          {image && (
            <div className="px-3 pt-3">
              <div className="relative inline-block">
                <img src={image} alt="Image to send" className="h-20 max-w-[12rem] rounded-lg border border-line object-cover" />
                <button
                  type="button"
                  onClick={() => setImage(null)}
                  aria-label="Remove image"
                  className="absolute -right-2 -top-2 flex size-6 items-center justify-center rounded-full border border-line-strong bg-surface text-ink-2 shadow hover:text-ink"
                >
                  <X className="size-3.5" />
                </button>
              </div>
            </div>
          )}

          <textarea
            ref={taRef}
            value={text}
            rows={1}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
            aria-label="Message"
            placeholder="Ask about your documents…"
            enterKeyHint={enterSends ? "send" : "enter"}
            className="block max-h-[200px] w-full resize-none bg-transparent px-4 pb-1 pt-3.5 leading-6 text-ink placeholder:text-ink-3 focus:outline-none"
          />

          <div className="flex items-center justify-between gap-2 px-2 pb-2">
            <div className="flex items-center gap-1">
              <input ref={fileRef} type="file" accept="image/*" className="sr-only" tabIndex={-1} onChange={(e) => { attach(e.target.files?.[0]); e.target.value = ""; }} />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                aria-label="Attach an image"
                title="Attach an image"
                className="flex size-9 items-center justify-center rounded-lg text-ink-2 transition hover:bg-sunken hover:text-ink"
              >
                <ImagePlus className="size-[18px]" />
              </button>
              <button
                type="button"
                onClick={() => setWebSearch((v) => !v)}
                aria-pressed={webSearch}
                title="Let Agentix search the web when your documents don’t cover a question"
                className={cn(
                  "flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] font-medium transition",
                  webSearch
                    ? "border-accent-line bg-accent-soft text-accent"
                    : "border-transparent text-ink-2 hover:bg-sunken hover:text-ink"
                )}
              >
                <Globe className="size-4" aria-hidden="true" />
                Web search
              </button>
            </div>

            {isStreaming ? (
              <button
                type="button"
                onClick={cancelStream}
                aria-label="Stop generating"
                className="flex size-9 items-center justify-center rounded-lg bg-ink text-bg transition hover:opacity-85"
              >
                <Square className="size-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!canSend}
                aria-label="Send message"
                className="flex size-9 items-center justify-center rounded-lg bg-accent text-accent-ink transition hover:brightness-110 disabled:bg-sunken disabled:text-ink-3"
              >
                <ArrowUp className="size-[18px]" />
              </button>
            )}
          </div>
        </div>

        <p className="mt-2 text-center text-[11px] text-ink-3">
          Answers are reviewed against sources but can still be wrong.
          {enterSends && <span className="hidden sm:inline"> · Enter to send, Shift + Enter for a new line</span>}
        </p>
      </div>
    </div>
  );
};
