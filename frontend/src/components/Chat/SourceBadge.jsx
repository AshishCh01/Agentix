import { FileText, ExternalLink, Globe } from "lucide-react";

export const SourceBadge = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-200/80 dark:border-slate-800/80 space-y-2">
      <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1">
        <span>Referenced Sources</span>
        <span className="text-slate-400 dark:text-slate-500">({sources.length})</span>
      </p>

      <div className="flex flex-wrap gap-2">
        {sources.map((src, idx) => {
          const isWeb = src.url || src.source_type === "web";
          const url = src.url;

          let displayTitle = `Source ${idx + 1}`;
          if (isWeb) {
            displayTitle = src.title || src.metadata?.source || src.filename || src.url || displayTitle;
          } else {
            const fileName = src.filename || src.title || src.metadata?.source || "Unknown Document";
            const pageStr = src.page_number ? ` (Page ${src.page_number})` : "";
            displayTitle = `${fileName}${pageStr}`;
          }

          // Sources carry no unique id from the API -- build a stable key
          // from their own identifying fields instead of the array index,
          // so React doesn't misattribute state/DOM across re-renders if
          // the list is ever reordered or filtered.
          const sourceKey = isWeb
            ? url || displayTitle
            : `${src.filename || "doc"}-${src.chunk_index ?? "x"}-${src.page_number ?? "x"}`;

          return isWeb && url ? (
            <a
              key={sourceKey}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 px-2.5 py-1 bg-white hover:bg-slate-50 border border-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700/80 dark:border-slate-700/60 rounded-lg text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors max-w-xs truncate"
            >
              <Globe className="w-3 h-3 flex-shrink-0" />
              <span className="truncate">{displayTitle}</span>
              <ExternalLink className="w-2.5 h-2.5 flex-shrink-0 opacity-70" />
            </a>
          ) : (
            <div
              key={sourceKey}
              className="inline-flex items-center space-x-1.5 px-2.5 py-1 bg-white border border-slate-200 dark:bg-slate-800/80 dark:border-slate-700/50 rounded-lg text-xs text-slate-600 dark:text-slate-300 max-w-xs truncate"
              title={src.content || displayTitle}
            >
              <FileText className="w-3 h-3 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              <span className="truncate">{displayTitle}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};