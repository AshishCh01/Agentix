import { memo, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { Check, Copy } from "lucide-react";
import "katex/dist/katex.min.css";
import { normalizeMath } from "../../utils/markdown";

const REMARK = [remarkGfm, remarkMath];
const REHYPE = [
  [rehypeKatex, { throwOnError: false, strict: false }],
  [rehypeHighlight, { detect: false, ignoreMissing: true }],
];

const nodeText = (node) => {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  return nodeText(node.props?.children);
};

const CodeBlock = ({ children }) => {
  const [copied, setCopied] = useState(false);
  const code = children?.props;
  const lang = /language-([\w-]+)/.exec(code?.className || "")?.[1];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(nodeText(children).replace(/\n$/, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard unavailable (insecure context) — nothing useful to do */
    }
  };

  return (
    <div className="codeblock">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="font-mono text-[11px] uppercase tracking-wider text-ink-3">{lang || "text"}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1.5 rounded px-1.5 py-1 text-xs text-ink-2 hover:text-ink"
        >
          {copied ? <Check className="size-3.5 text-accent" /> : <Copy className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
};

const COMPONENTS = {
  pre: CodeBlock,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
};

/** Renders assistant output: GitHub-flavoured Markdown, LaTeX math and highlighted code. */
export const Markdown = memo(function Markdown({ content }) {
  const text = useMemo(() => normalizeMath(content), [content]);
  return (
    <div className="prose-agentix">
      <ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE} components={COMPONENTS}>
        {text}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;
