import { LogoMark } from "../ui/Logo";

const STARTERS = [
  "Summarize the main points of my documents",
  "List the key terms and what they mean",
  "Explain this to me as if I’m new to the topic",
];

export const EmptyState = ({ scopeName, onPick, disabled }) => (
  <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center px-5 py-10 text-center">
    <LogoMark className="mb-5 h-12" />
    <h2 className="font-display text-3xl leading-tight text-ink sm:text-4xl">What would you like to know?</h2>
    <p className="mt-2.5 max-w-md text-sm leading-relaxed text-ink-2">
      {scopeName
        ? `Questions in this chat are answered from “${scopeName}”.`
        : "Add a document to this chat, then ask about it. Answers come with their sources."}
    </p>
    <div className="mt-7 flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:flex-wrap sm:justify-center">
      {STARTERS.map((text) => (
        <button
          key={text}
          type="button"
          disabled={disabled}
          onClick={() => onPick(text)}
          className="rounded-full border border-line-strong bg-surface px-4 py-2 text-left text-[13px] text-ink-2 transition hover:border-accent hover:text-ink disabled:opacity-50"
        >
          {text}
        </button>
      ))}
    </div>
  </div>
);
