import { cn } from "../../utils/cn";

/** Centered empty / error message used across pages. */
export const StateMessage = ({ icon: Icon, title, description, action, tone = "neutral", className }) => (
  <div className={cn("mx-auto flex max-w-sm flex-col items-center px-4 py-16 text-center", className)}>
    {Icon && (
      <div
        className={cn(
          "mb-4 flex size-11 items-center justify-center rounded-xl border",
          tone === "danger"
            ? "border-danger/30 bg-danger-soft text-danger"
            : "border-accent-line bg-accent-soft text-accent"
        )}
      >
        <Icon className="size-5" aria-hidden="true" />
      </div>
    )}
    <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
    {description && <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{description}</p>}
    {action && <div className="mt-5">{action}</div>}
  </div>
);
