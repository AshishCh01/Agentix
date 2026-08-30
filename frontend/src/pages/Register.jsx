import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { Spinner } from "../components/Common/Loader";
import { getErrorMessage } from "../utils/helpers";

export const Register = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { signup } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await signup(email, password);
      navigate("/");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="p-3 text-xs rounded-lg bg-red-50 border border-red-200 text-red-600 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400">
          {error}
        </div>
      )}

      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
          Email Address
        </label>
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="developer@example.com"
          className="w-full px-3 py-2 bg-white border border-slate-300 dark:bg-slate-950 dark:border-slate-800 rounded-lg text-sm text-slate-900 placeholder-slate-400 dark:text-slate-100 dark:placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
          Password
        </label>
        <input
          type="password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="••••••••"
          className="w-full px-3 py-2 bg-white border border-slate-300 dark:bg-slate-950 dark:border-slate-800 rounded-lg text-sm text-slate-900 placeholder-slate-400 dark:text-slate-100 dark:placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
          Confirm Password
        </label>
        <input
          type="password"
          required
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          placeholder="••••••••"
          className="w-full px-3 py-2 bg-white border border-slate-300 dark:bg-slate-950 dark:border-slate-800 rounded-lg text-sm text-slate-900 placeholder-slate-400 dark:text-slate-100 dark:placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-500 font-medium text-sm text-white rounded-lg transition-colors flex items-center justify-center disabled:opacity-50"
      >
        {loading ? <Spinner size="sm" /> : "Create Account"}
      </button>

      <div className="text-center text-xs text-slate-500 dark:text-slate-400 pt-2">
        Already have an account?{" "}
        <Link to="/login" className="text-blue-600 dark:text-blue-400 hover:underline font-medium">
          Sign In
        </Link>
      </div>
    </form>
  );
};