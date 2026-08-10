import { Link } from "react-router-dom";

export const NotFound = () => {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center p-4 text-center">
      <h1 className="text-6xl font-bold text-blue-500">404</h1>
      <p className="mt-2 text-lg text-slate-300">Page Not Found</p>
      <Link
        to="/"
        className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm transition-colors"
      >
        Return to Workspace
      </Link>
    </div>
  );
};