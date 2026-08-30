import { Outlet } from "react-router-dom";
import { ThemeToggle } from "../components/Common/ThemeToggle";

export const AuthLayout = () => {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex flex-col justify-center items-center p-4 relative">
      <ThemeToggle className="absolute top-4 right-4" />

      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <img
            src="/Agentix_logo_mark.png"
            alt="Agentix"
            className="h-11 w-auto mx-auto mb-2"
          />
          <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Agentix
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Multi-Agent Document Intelligence
          </p>
        </div>

        <div className="bg-white border border-slate-200 dark:bg-slate-900 dark:border-slate-800 rounded-2xl p-6 shadow-xl dark:shadow-2xl">
          <Outlet />
        </div>
      </div>
    </div>
  );
};