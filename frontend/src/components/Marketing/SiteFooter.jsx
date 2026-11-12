import { Logo } from "../ui/Logo";

export const SiteFooter = () => (
  <footer className="border-t border-line">
    <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-8 pb-safe sm:flex-row sm:items-center sm:justify-between sm:px-8">
      <Logo markClassName="h-6" />
      <div className="flex items-center gap-6 text-sm text-ink-3">
        <a
          href="https://github.com/AshishCh01/Agentix"
          target="_blank"
          rel="noopener noreferrer"
          className="transition hover:text-ink"
        >
          Source on GitHub
        </a>
        <span>© {new Date().getFullYear()} Agentix</span>
      </div>
    </div>
  </footer>
);
