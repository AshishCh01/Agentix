import { Loader2 } from "lucide-react";
import { cn } from "../../utils/cn";
import { LogoMark } from "../ui/Logo";

const SIZES = { sm: "size-4", md: "size-5", lg: "size-7" };

export const Spinner = ({ size = "md", className }) => (
  <Loader2 aria-hidden="true" className={cn("animate-spin text-accent", SIZES[size] || SIZES.md, className)} />
);

export const FullPageLoader = () => (
  <div role="status" aria-label="Loading" className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-bg">
    <LogoMark className="h-10 animate-pulse" />
    <span className="sr-only">Loading…</span>
  </div>
);
