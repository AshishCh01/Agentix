import { Loader2 } from "lucide-react";
import { cn } from "../../utils/cn";

const VARIANTS = {
  primary: "bg-accent text-accent-ink hover:brightness-110 active:brightness-95",
  secondary: "bg-surface text-ink border border-line-strong hover:bg-sunken",
  ghost: "text-ink-2 hover:text-ink hover:bg-sunken",
  danger: "bg-danger text-white hover:brightness-110",
  inverse: "bg-bg text-ink hover:brightness-95",
};
const SIZES = {
  sm: "h-8 px-3 text-[13px] gap-1.5",
  md: "h-10 px-4 text-sm gap-2",
  lg: "h-12 px-6 text-[15px] gap-2",
};

export const Button = ({
  as: Comp = "button",
  variant = "primary",
  size = "md",
  loading = false,
  className,
  children,
  disabled,
  ...props
}) => (
  <Comp
    {...(Comp === "button" ? { type: props.type || "button", disabled: disabled || loading } : {})}
    aria-busy={loading || undefined}
    className={cn(
      "inline-flex shrink-0 items-center justify-center rounded-lg font-medium transition select-none",
      "disabled:pointer-events-none disabled:opacity-50",
      VARIANTS[variant],
      SIZES[size],
      className
    )}
    {...props}
  >
    {loading && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
    {children}
  </Comp>
);

export const IconButton = ({ label, className, children, size = "md", ...props }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    className={cn(
      "inline-flex shrink-0 items-center justify-center rounded-lg text-ink-2 transition",
      "hover:bg-sunken hover:text-ink disabled:pointer-events-none disabled:opacity-50",
      size === "sm" ? "size-8" : "size-10",
      className
    )}
    {...props}
  >
    {children}
  </button>
);
