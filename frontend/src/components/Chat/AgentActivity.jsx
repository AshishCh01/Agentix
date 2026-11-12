import { Brain, CheckCheck, Database, Globe, PenLine, Sparkles } from "lucide-react";
import { AGENT_NODES, AGENT_NODE_LABELS } from "../../utils/constants";

const ICONS = {
  [AGENT_NODES.SUPERVISOR]: Brain,
  [AGENT_NODES.VECTOR_SEARCH]: Database,
  [AGENT_NODES.WEB_SEARCH]: Globe,
  [AGENT_NODES.ANSWER]: PenLine,
  [AGENT_NODES.REFLECTION]: CheckCheck,
  [AGENT_NODES.GREETING]: Sparkles,
};

/** Live "what is the agent doing" line shown while a response streams. */
export const AgentActivity = ({ node }) => {
  const Icon = ICONS[node] || Sparkles;
  const label = AGENT_NODE_LABELS[node] || "Working";
  return (
    <div role="status" aria-live="polite" className="inline-flex items-center gap-2 text-[13px] text-ink-2">
      <Icon className="size-4 text-accent" aria-hidden="true" />
      <span>{label}</span>
      <span className="inline-flex gap-0.5" aria-hidden="true">
        <i className="dot-1 size-1 rounded-full bg-current" />
        <i className="dot-2 size-1 rounded-full bg-current" />
        <i className="dot-3 size-1 rounded-full bg-current" />
      </span>
    </div>
  );
};
