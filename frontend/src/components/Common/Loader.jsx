import { Loader2 } from "lucide-react";

export const Spinner = ({ size = "md", className = "" }) => {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
    xl: "w-12 h-12",
  };

  return (
    <Loader2
      className={`animate-spin text-blue-600 dark:text-blue-500 ${sizeClasses[size] || sizeClasses.md} ${className}`}
    />
  );
};

export const FullPageLoader = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <Spinner size="xl" />
      <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 font-medium animate-pulse">
        Initializing Agentix Platform...
      </p>
    </div>
  );
};