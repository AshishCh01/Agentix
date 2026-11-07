import { memo } from "react";
import { CheckCircle2, AlertTriangle } from "lucide-react";
import { getVerificationBadgeConfig } from "../../utils/chatHelpers";

export const VerificationBadge = memo(function VerificationBadge({ message }) {
  const config = getVerificationBadgeConfig(message);
  if (!config) return null;

  if (config.type === "verified") {
    return (
      <span
        role="status"
        aria-label={config.ariaLabel}
        title={config.title}
        className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium text-emerald-700 bg-emerald-50 border border-emerald-200/80 dark:text-emerald-400 dark:bg-emerald-950/40 dark:border-emerald-800/60 rounded-full"
      >
        <CheckCircle2
          className="w-3 h-3 text-emerald-600 dark:text-emerald-400 flex-shrink-0"
          aria-hidden="true"
        />
        <span>{config.label}</span>
      </span>
    );
  }

  if (config.type === "unverified") {
    return (
      <span
        role="status"
        aria-label={config.ariaLabel}
        title={config.title}
        className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium text-amber-700 bg-amber-50 border border-amber-200/80 dark:text-amber-400 dark:bg-amber-950/40 dark:border-amber-800/60 rounded-full"
      >
        <AlertTriangle
          className="w-3 h-3 text-amber-600 dark:text-amber-400 flex-shrink-0"
          aria-hidden="true"
        />
        <span>{config.label}</span>
      </span>
    );
  }

  return null;
});
