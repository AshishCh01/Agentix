import { Check, FileText, ShieldCheck } from "lucide-react";
import { LogoMark } from "../ui/Logo";

const STEPS = ["Understood the question", "Searched your document index", "Wrote an answer from the passages", "Checked it against the sources"];

/** A static illustration of one finished exchange, built from real UI pieces. */
export const HeroPreview = () => (
  <figure aria-label="Example conversation" className="animate-fade-up [animation-delay:120ms]">
    <div className="overflow-hidden rounded-2xl border border-line bg-surface shadow-[0_30px_70px_-30px_rgba(20,32,29,0.35)]">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <p className="text-[13px] font-semibold text-ink">Optimization notes</p>
          <p className="text-xs text-ink-3">1 document</p>
        </div>
        <span className="label-mono">Example</span>
      </div>

      <div className="space-y-6 p-4 sm:p-5">
        <div className="flex justify-end">
          <p className="max-w-[85%] rounded-2xl rounded-br-md border border-accent-line bg-accent-soft px-4 py-2.5 text-[15px] text-ink">
            What convergence rate does subgradient descent get?
          </p>
        </div>

        <div className="flex gap-3">
          <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border border-line bg-surface">
            <LogoMark className="h-4" />
          </div>
          <div className="min-w-0 flex-1 space-y-4">
            <ol className="space-y-1.5">
              {STEPS.map((s) => (
                <li key={s} className="flex items-center gap-2 text-[13px] text-ink-2">
                  <span className="flex size-4 items-center justify-center rounded-full bg-accent-soft text-accent">
                    <Check className="size-3" strokeWidth={3} aria-hidden="true" />
                  </span>
                  {s}
                </li>
              ))}
            </ol>
            <p className="text-[15px] leading-relaxed text-ink">
              For non-smooth convex objectives, subgradient descent converges at a rate of{" "}
              <code className="rounded border border-line bg-sunken px-1.5 py-0.5 font-mono text-[0.8em]">O(1/√k)</code>, which is slower than the{" "}
              <code className="rounded border border-line bg-sunken px-1.5 py-0.5 font-mono text-[0.8em]">O(1/k)</code> rate of gradient descent on smooth functions.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink-2">
                <FileText className="size-3.5 text-accent" aria-hidden="true" />
                optimization-notes.pdf · p.14
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-accent-line bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent">
                <ShieldCheck className="size-3.5" aria-hidden="true" />
                Verified against your documents
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </figure>
);
