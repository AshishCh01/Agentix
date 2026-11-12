import { Link } from "react-router-dom";
import { cn } from "../../utils/cn";

export const LogoMark = ({ className }) => (
  <img src="/Agentix_logo_mark.png" alt="" aria-hidden="true" className={cn("h-7 w-auto", className)} />
);

export const Logo = ({ to = "/", className, markClassName }) => (
  <Link to={to} aria-label="Agentix home" className={cn("inline-flex items-center gap-2.5", className)}>
    <LogoMark className={markClassName} />
    <span className="text-[17px] font-semibold tracking-tight text-ink">Agentix</span>
  </Link>
);
