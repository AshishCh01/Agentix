import { memo } from "react";
import { ShieldCheck, TriangleAlert } from "lucide-react";
import { getVerificationBadgeConfig } from "../../utils/chatHelpers";
import { cn } from "../../utils/cn";

export const VerificationBadge = memo(function VerificationBadge({ message }) {
  const config = getVerificationBadgeConfig(message);
  if (!config) return null;
  const ok = config.type === "verified";
  const Icon = ok ? ShieldCheck : TriangleAlert;

  return (
    <span
      role="status"
      aria-label={config.ariaLabel}
      title={config.title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        ok ? "border-accent-line bg-accent-soft text-accent" : "border-warn/30 bg-warn-soft text-warn"
      )}
    >
      <Icon className="size-3.5 shrink-0" aria-hidden="true" />
      {config.label}
    </span>
  );
});
