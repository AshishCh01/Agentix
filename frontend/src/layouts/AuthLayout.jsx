import { Outlet } from "react-router-dom";
import { Bot } from "lucide-react";

export const AuthLayout = () => {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center p-4">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-blue-600/10 border border-blue-500/20 text-blue-400 mb-2">
            <Bot className="w-7 h-7" />
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">
            Agentic RAG Studio
          </h2>
          <p className="text-sm text-slate-400">
            Enterprise Multi-Agent Document Intelligence
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl">
          <Outlet />
        </div>
      </div>
    </div>
  );
};