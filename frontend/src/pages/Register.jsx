import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { MailCheck } from "lucide-react";
import { useAuth } from "../hooks/useAuth";
import { usePageTitle } from "../hooks/usePageTitle";
import { Button } from "../components/ui/Button";
import { Field, Input } from "../components/ui/Field";
import { PasswordInput } from "../components/ui/PasswordInput";
import { StateMessage } from "../components/ui/StateMessage";

const MIN_PASSWORD = 8;

export default function Register() {
  usePageTitle("Create account");
  const { signup } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [needsConfirmation, setNeedsConfirmation] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (password.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters for your password.`);
      return;
    }
    setError("");
    setLoading(true);
    try {
      const data = await signup(email.trim(), password);
      // With email confirmation enabled there is no session until the link is clicked.
      if (data?.session) {
        navigate("/app", { replace: true });
      } else {
        setNeedsConfirmation(true);
        setLoading(false);
      }
    } catch (err) {
      setError(err?.message || "We couldn’t create your account. Please try again.");
      setLoading(false);
    }
  };

  if (needsConfirmation) {
    return (
      <StateMessage
        className="py-0"
        icon={MailCheck}
        title="Check your email"
        description={`We sent a confirmation link to ${email}. Open it, then come back to sign in.`}
        action={<Button as={Link} to="/login" variant="secondary">Go to sign in</Button>}
      />
    );
  }

  return (
    <div className="animate-fade-up">
      <h1 className="font-display text-4xl leading-tight text-ink">Create your account</h1>
      <p className="mt-2 text-sm text-ink-2">Upload a document and ask your first question in a minute.</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-4">
        <Field label="Email">
          {(p) => (
            <Input {...p} type="email" required autoComplete="email" inputMode="email" autoFocus
              placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
          )}
        </Field>
        <Field label="Password" hint={`At least ${MIN_PASSWORD} characters.`}>
          {(p) => (
            <PasswordInput {...p} required autoComplete="new-password" minLength={MIN_PASSWORD}
              placeholder="Create a password" value={password} onChange={(e) => setPassword(e.target.value)} />
          )}
        </Field>

        {error && (
          <p role="alert" className="rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm text-danger">{error}</p>
        )}

        <Button type="submit" size="lg" loading={loading} className="w-full">Create account</Button>
      </form>

      <p className="mt-6 text-center text-sm text-ink-2">
        Already have an account?{" "}
        <Link to="/login" state={location.state} className="font-medium text-accent underline underline-offset-4">Sign in</Link>
      </p>
    </div>
  );
}
