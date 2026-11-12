import { Link } from "react-router-dom";
import { ArrowRight, Binary, BookMarked, CheckCheck, Database, Globe, ImagePlus, PenLine, Quote, Brain, ShieldCheck } from "lucide-react";
import { useAuth } from "../hooks/useAuth";
import { usePageTitle } from "../hooks/usePageTitle";
import { Button } from "../components/ui/Button";
import { SiteHeader } from "../components/Marketing/SiteHeader";
import { SiteFooter } from "../components/Marketing/SiteFooter";
import { HeroPreview } from "../components/Marketing/HeroPreview";
import { MAX_UPLOAD_MB } from "../utils/files";

const PIPELINE = [
  { icon: Brain, title: "Understand", text: "A supervisor agent reads your question and decides whether it needs your documents, the web, or just a reply." },
  { icon: Database, title: "Retrieve", text: "It searches the index built from your files. The web is only used if you turn web search on." },
  { icon: PenLine, title: "Answer", text: "The answer streams in as it is written, with the passages it drew on attached as sources." },
  { icon: CheckCheck, title: "Verify", text: "A review step checks the draft against those passages. You see whether it passed, or that it could not be fully verified." },
];

const FEATURES = [
  { icon: BookMarked, title: "Knowledge bases", text: "Group documents by course, project or client, then point a chat at just that set." },
  { icon: Quote, title: "Cited sources", text: "Each answer lists the files and pages it came from, so you can go back to the original." },
  { icon: ShieldCheck, title: "Reviewed answers", text: "A separate review step checks every draft. A badge tells you the result instead of hiding it." },
  { icon: Globe, title: "Web search, on your terms", text: "When your documents don’t cover a question, Agentix says so and offers a web search rather than guessing." },
  { icon: ImagePlus, title: "Ask about images", text: "Attach a screenshot, diagram or photo of a page and ask about it in the same chat." },
  { icon: Binary, title: "Math and code that render", text: "Formulas are typeset, tables stay readable, and code gets highlighting with a copy button." },
];

export default function Landing() {
  usePageTitle();
  const { isAuthenticated } = useAuth();
  const ctaTo = isAuthenticated ? "/app" : "/register";
  const ctaLabel = isAuthenticated ? "Open workspace" : "Get started";

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:shadow-lg">
        Skip to content
      </a>
      <SiteHeader />

      <main id="main">
        {/* Hero */}
        <section className="mx-auto grid max-w-6xl items-center gap-12 px-5 pb-20 pt-12 sm:px-8 sm:pt-16 lg:grid-cols-[1.05fr_1fr] lg:gap-14 lg:pb-28 lg:pt-24">
          <div className="animate-fade-up">
            <p className="label-mono mb-5">Multi-agent document Q&amp;A</p>
            <h1 className="font-display text-[3.25rem] leading-[0.98] tracking-tight text-ink sm:text-7xl">
              Ask your documents. Get answers you can <em className="text-accent">check</em>.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
              Agentix reads the files you upload, shows where each answer came from, and has a second agent review the
              answer against those sources before you rely on it.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Button as={Link} to={ctaTo} size="lg">
                {ctaLabel}
                <ArrowRight className="size-4" aria-hidden="true" />
              </Button>
              <Button as="a" href="#how-it-works" variant="secondary" size="lg">See how it works</Button>
            </div>
            <p className="mt-5 text-sm text-ink-3">PDF, Word, Excel, CSV, text, Markdown, JSON and images · up to {MAX_UPLOAD_MB} MB per file</p>
          </div>
          <HeroPreview />
        </section>

        {/* How it works */}
        <section id="how-it-works" className="scroll-mt-20 border-y border-line bg-sunken">
          <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 lg:py-24">
            <p className="label-mono mb-3">How an answer is made</p>
            <h2 className="max-w-2xl font-display text-4xl leading-tight text-ink sm:text-5xl">
              Four steps between your question and the answer.
            </h2>
            <ol className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
              {PIPELINE.map(({ icon: Icon, title, text }, i) => (
                <li key={title} className="bg-surface p-6">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-ink-3">0{i + 1}</span>
                    <Icon className="size-5 text-accent" aria-hidden="true" />
                  </div>
                  <h3 className="mt-8 text-lg font-semibold text-ink">{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-ink-2">{text}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="mx-auto max-w-6xl scroll-mt-20 px-5 py-20 sm:px-8 lg:py-28">
          <p className="label-mono mb-3">What you get</p>
          <h2 className="max-w-2xl font-display text-4xl leading-tight text-ink sm:text-5xl">
            Built for studying, research and reading closely.
          </h2>
          <dl className="mt-12 grid gap-x-10 gap-y-10 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <div key={title} className="border-t border-line-strong pt-5">
                <dt className="flex items-center gap-2.5 text-[15px] font-semibold text-ink">
                  <Icon className="size-[18px] text-accent" aria-hidden="true" />
                  {title}
                </dt>
                <dd className="mt-2 text-sm leading-relaxed text-ink-2">{text}</dd>
              </div>
            ))}
          </dl>
        </section>

        {/* Closing CTA */}
        <section className="px-5 pb-20 sm:px-8 lg:pb-28">
          <div className="mx-auto max-w-6xl rounded-3xl bg-ink px-6 py-14 text-bg sm:px-12 sm:py-20">
            <h2 className="max-w-2xl font-display text-4xl leading-tight sm:text-6xl">
              Bring a document. Ask the first question.
            </h2>
            <div className="mt-8">
              <Button as={Link} to={ctaTo} variant="inverse" size="lg">
                {ctaLabel}
                <ArrowRight className="size-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
