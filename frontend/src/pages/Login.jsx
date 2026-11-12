import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { usePageTitle } from "../hooks/usePageTitle";
import { Button } from "../components/ui/Button";
import { Field, Input } from "../components/ui/Field";
import { PasswordInput } from "../components/ui/PasswordInput";

export default function Login() {
  usePageTitle("Sign in");
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    setError("");
    setLoading(true);
    try {
      await login(email.trim(), password);
      navigate(location.state?.from || "/app", { replace: true });
    } catch (err) {
      setError(err?.message || "We couldn’t sign you in. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-up">
      <h1 className="font-display text-4xl leading-tight text-ink">Welcome back</h1>
      <p className="mt-2 text-sm text-ink-2">Sign in to continue to your workspace.</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-4" noValidate={false}>
        <Field label="Email">
          {(p) => (
            <Input {...p} type="email" required autoComplete="email" inputMode="email" autoFocus
              placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
          )}
        </Field>
        <Field label="Password">
          {(p) => (
            <PasswordInput {...p} required autoComplete="current-password" placeholder="Your password"
              value={password} onChange={(e) => setPassword(e.target.value)} />
          )}
        </Field>

        {error && (
          <p role="alert" className="rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger">{error}</p>
        )}

        <Button type="submit" size="lg" loading={loading} className="w-full">Sign in</Button>
      </form>

      <p className="mt-6 text-center text-sm text-ink-2">
        New to Agentix?{" "}
        <Link to="/register" state={location.state} className="font-medium text-accent underline underline-offset-4">
          Create an account
        </Link>
      </p>
    </div>
  );
}
