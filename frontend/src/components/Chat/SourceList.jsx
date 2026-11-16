import { useState } from "react";
import { ExternalLink, FileText, Globe } from "lucide-react";
import { cn } from "../../utils/cn";

const VISIBLE = 3;

const describe = (src, idx) => {
  const isWeb = Boolean(src.url) || src.source_type === "web";
  if (isWeb) {
    return {
      isWeb,
      title: src.title || src.metadata?.source || src.filename || src.url || `Source ${idx + 1}`,
      key: src.url || `web-${idx}`,
    };
  }
  const file = src.filename || src.title || src.metadata?.source || "Document";
  return {
    isWeb,
    title: src.page_number ? `${file} · p.${src.page_number}` : file,
    // Sources carry no id from the API; build a stable key from identifying fields.
    key: `${src.filename || "doc"}-${src.chunk_index ?? "x"}-${src.page_number ?? "x"}`,
  };
};

const chip =
  "inline-flex max-w-full items-center gap-1.5 rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink-2";

export const SourceList = ({ sources }) => {
  const [expanded, setExpanded] = useState(false);
  if (!sources?.length) return null;

  const shown = expanded ? sources : sources.slice(0, VISIBLE);
  const hidden = sources.length - shown.length;

  return (
    <div className="space-y-2">
      <p className="label-mono">Sources · {sources.length}</p>
      <ul className="flex flex-wrap gap-1.5">
        {shown.map((src, idx) => {
          const d = describe(src, idx);
          const Icon = d.isWeb ? Globe : FileText;
          return (
            <li key={d.key} className="min-w-0 max-w-full">
              {d.isWeb && src.url ? (
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={cn(chip, "hover:border-accent-line hover:text-accent")}
                >
                  <Icon className="size-3.5 shrink-0" aria-hidden="true" />
                  <span className="truncate">{d.title}</span>
                  <ExternalLink className="size-3 shrink-0 opacity-60" aria-hidden="true" />
                </a>
              ) : (
                <span className={chip} title={src.content || d.title}>
                  <Icon className="size-3.5 shrink-0 text-accent" aria-hidden="true" />
                  <span className="truncate">{d.title}</span>
                </span>
              )}
            </li>
          );
        })}
        {hidden > 0 && (
          <li>
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="rounded-md px-2 py-1 text-xs font-medium text-accent hover:underline"
            >
              +{hidden} more
            </button>
          </li>
        )}
      </ul>
    </div>
  );
};
