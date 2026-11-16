import { Outlet } from "react-router-dom";
import { ShieldCheck } from "lucide-react";
import { Logo } from "../components/ui/Logo";
import { ThemeToggle } from "../components/Common/ThemeToggle";

export default function AuthLayout() {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <div className="flex flex-col px-5 pb-safe pt-safe sm:px-10">
        <header className="flex h-16 items-center justify-between">
          <Logo />
          <ThemeToggle />
        </header>
        <main id="main" className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center py-10">
          <Outlet />
        </main>
      </div>

      <aside className="relative hidden overflow-hidden border-l border-line bg-sunken lg:flex lg:flex-col lg:justify-end lg:p-14">
        <img
          src="/Agentix_logo_mark.png"
          alt=""
          aria-hidden="true"
          className="pointer-events-none absolute -right-10 -top-6 h-[26rem] w-auto opacity-[0.14]"
        />
        <div className="relative max-w-md">
          <p className="label-mono mb-4">Multi-agent document Q&amp;A</p>
          <p className="font-display text-5xl leading-[1.05] text-ink">
            Answers you can <em className="text-accent">check</em>, not just trust.
          </p>
          <p className="mt-5 flex items-start gap-2.5 text-sm leading-relaxed text-ink-2">
            <ShieldCheck className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden="true" />
            Every answer lists its sources and is reviewed against them before it reaches you.
          </p>
        </div>
      </aside>
    </div>
  );
}
