import { useState } from "react";
import { Link } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { useAuth } from "../../hooks/useAuth";
import { Button, IconButton } from "../ui/Button";
import { Logo } from "../ui/Logo";
import { ThemeToggle } from "../Common/ThemeToggle";

const LINKS = [
  { href: "#how-it-works", label: "How it works" },
  { href: "#features", label: "Features" },
];

export const SiteHeader = () => {
  const { isAuthenticated } = useAuth();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 pt-safe backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
        <Logo />

        <nav aria-label="Main" className="hidden items-center gap-7 md:flex">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="text-sm text-ink-2 transition hover:text-ink">{l.label}</a>
          ))}
        </nav>

        <div className="flex items-center gap-1.5">
          <ThemeToggle />
          {isAuthenticated ? (
            <Button as={Link} to="/app" size="sm">Open workspace</Button>
          ) : (
            <>
              <Button as={Link} to="/login" variant="ghost" size="sm" className="hidden sm:inline-flex">Sign in</Button>
              <Button as={Link} to="/register" size="sm" className="hidden sm:inline-flex">Get started</Button>
            </>
          )}
          <IconButton
            label={open ? "Close menu" : "Open menu"}
            className="md:hidden"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </IconButton>
        </div>
      </div>

      {open && (
        <nav aria-label="Mobile" className="animate-fade-up border-t border-line bg-bg px-5 py-3 md:hidden">
          <ul className="space-y-1">
            {LINKS.map((l) => (
              <li key={l.href}>
                <a href={l.href} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-3 text-[15px] text-ink hover:bg-sunken">
                  {l.label}
                </a>
              </li>
            ))}
            {!isAuthenticated && (
              <li className="grid grid-cols-2 gap-2 pt-2">
                <Button as={Link} to="/login" variant="secondary">Sign in</Button>
                <Button as={Link} to="/register">Get started</Button>
              </li>
            )}
          </ul>
        </nav>
      )}
    </header>
  );
};
