import { memo } from "react";
import { User } from "lucide-react";
import { SourceBadge } from "./SourceBadge";
import { formatDate } from "../../utils/helpers";

// Memoized so that during token streaming -- where ChatContext rebuilds the
// `messages` array on every token but only replaces the last message's
// object -- every other message keeps the same `message` prop reference and
// skips re-rendering entirely, instead of the whole list re-rendering per
// token.
export const MessageItem = memo(function MessageItem({ message }) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex space-x-3 max-w-4xl mx-auto ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-blue-50 border border-blue-100 dark:bg-blue-600/10 dark:border-blue-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
          <img src="/Agentix_logo_mark.png" alt="Agentix" className="w-4 h-4 object-contain" />
        </div>
      )}

      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl p-4 text-xs sm:text-sm shadow-sm dark:shadow-md leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white rounded-tr-none"
            : "bg-slate-50 border border-slate-200 text-slate-700 dark:bg-slate-950 dark:border-slate-800 dark:text-slate-100 rounded-tl-none"
        }`}
      >
        <div className="flex items-center justify-between space-x-4 mb-1.5 text-[10px] opacity-70">
          <span className="font-semibold uppercase tracking-wider">
            {isUser ? "You" : "Assistant"}
          </span>
          <span>{formatDate(message.created_at)}</span>
        </div>

        {message.image_data && (
          <div className="mb-2.5">
            <img
              src={message.image_data}
              alt="User attachment"
              className="max-w-full max-h-60 rounded-xl border border-black/10 dark:border-white/20 object-cover shadow"
            />
          </div>
        )}

        {message.content && (
          <div className="whitespace-pre-wrap break-words">{message.content}</div>
        )}

        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceBadge sources={message.sources} />
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-slate-100 border border-slate-200 text-slate-600 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  );
});