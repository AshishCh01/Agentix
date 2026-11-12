import { useId } from "react";
import { cn } from "../../utils/cn";

const control =
  "w-full rounded-lg border border-line-strong bg-surface px-3 text-ink placeholder:text-ink-3 " +
  "transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:opacity-60";

export const Input = ({ className, ...props }) => (
  <input className={cn(control, "h-11", className)} {...props} />
);

export const Textarea = ({ className, ...props }) => (
  <textarea className={cn(control, "py-2.5 leading-relaxed", className)} {...props} />
);

/** Label + control + hint/error, wired together for screen readers. */
export const Field = ({ label, hint, error, optional, children }) => {
  const id = useId();
  const describedBy = error ? `${id}-err` : hint ? `${id}-hint` : undefined;
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="flex items-baseline justify-between text-[13px] font-medium text-ink">
        {label}
        {optional && <span className="text-xs font-normal text-ink-3">Optional</span>}
      </label>
      {children({ id, "aria-describedby": describedBy, "aria-invalid": error ? true : undefined })}
      {error ? (
        <p id={`${id}-err`} role="alert" className="text-xs text-danger">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs text-ink-3">{hint}</p>
      ) : null}
    </div>
  );
};
