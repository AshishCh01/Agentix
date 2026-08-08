import React from "react";
import { FileText, ExternalLink, Globe } from "lucide-react";

export const SourceBadge = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-2">
      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
        <span>Referenced Sources</span>
        <span className="text-slate-500">({sources.length})</span>
      </p>

      <div className="flex flex-wrap gap-2">
        {sources.map((src, idx) => {
          const isWeb = src.url || src.source_type === "web";
          const title = src.title || src.metadata?.source || `Source ${idx + 1}`;
          const url = src.url;

          return isWeb && url ? (
            <a
              key={idx}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700/80 border border-slate-700/60 rounded-lg text-xs text-blue-400 hover:text-blue-300 transition-colors max-w-xs truncate"
            >
              <Globe className="w-3 h-3 flex-shrink-0" />
              <span className="truncate">{title}</span>
              <ExternalLink className="w-2.5 h-2.5 flex-shrink-0 opacity-70" />
            </a>
          ) : (
            <div
              key={idx}
              className="inline-flex items-center space-x-1.5 px-2.5 py-1 bg-slate-800/80 border border-slate-700/50 rounded-lg text-xs text-slate-300 max-w-xs truncate"
              title={src.content || title}
            >
              <FileText className="w-3 h-3 text-emerald-400 flex-shrink-0" />
              <span className="truncate">{title}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};