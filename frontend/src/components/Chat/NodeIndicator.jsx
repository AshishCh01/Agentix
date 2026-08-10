import { Cpu, Database, Globe, Sparkles, CheckCircle, Bot } from "lucide-react";
import { AGENT_NODES, AGENT_NODE_LABELS } from "../../utils/constants";

export const NodeIndicator = ({ node }) => {
  if (!node) return null;

  const getNodeConfig = (nodeName) => {
    switch (nodeName) {
      case AGENT_NODES.SUPERVISOR:
        return {
          icon: <Cpu className="w-3.5 h-3.5 text-purple-400 animate-spin" />,
          bgColor: "bg-purple-500/10",
          borderColor: "border-purple-500/20",
          textColor: "text-purple-300",
        };
      case AGENT_NODES.VECTOR_SEARCH:
        return {
          icon: <Database className="w-3.5 h-3.5 text-blue-400 animate-pulse" />,
          bgColor: "bg-blue-500/10",
          borderColor: "border-blue-500/20",
          textColor: "text-blue-300",
        };
      case AGENT_NODES.WEB_SEARCH:
        return {
          icon: <Globe className="w-3.5 h-3.5 text-emerald-400 animate-bounce" />,
          bgColor: "bg-emerald-500/10",
          borderColor: "border-emerald-500/20",
          textColor: "text-emerald-300",
        };
      case AGENT_NODES.ANSWER:
        return {
          icon: <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-pulse" />,
          bgColor: "bg-amber-500/10",
          borderColor: "border-amber-500/20",
          textColor: "text-amber-300",
        };
      case AGENT_NODES.REFLECTION:
        return {
          icon: <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />,
          bgColor: "bg-cyan-500/10",
          borderColor: "border-cyan-500/20",
          textColor: "text-cyan-300",
        };
      default:
        return {
          icon: <Bot className="w-3.5 h-3.5 text-slate-400 animate-pulse" />,
          bgColor: "bg-slate-800",
          borderColor: "border-slate-700",
          textColor: "text-slate-300",
        };
    }
  };

  const config = getNodeConfig(node);
  const label = AGENT_NODE_LABELS[node] || "Processing Query...";

  return (
    <div className="flex items-center space-x-2 my-2">
      <div
        className={`inline-flex items-center space-x-2 px-3 py-1.5 rounded-full border ${config.bgColor} ${config.borderColor} ${config.textColor} text-xs font-medium backdrop-blur-sm shadow-sm`}
      >
        {config.icon}
        <span>{label}</span>
      </div>
    </div>
  );
};