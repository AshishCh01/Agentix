import { Link } from "react-router-dom";
import { Button } from "../components/ui/Button";
import { Logo } from "../components/ui/Logo";
import { usePageTitle } from "../hooks/usePageTitle";

export default function NotFound() {
  usePageTitle("Page not found");
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-6 px-6 text-center">
      <Logo />
      <div>
        <p className="label-mono">Error 404</p>
        <h1 className="mt-2 font-display text-5xl text-ink">That page doesn’t exist</h1>
        <p className="mt-3 text-sm text-ink-2">The link may be broken, or the page may have moved.</p>
      </div>
      <Button as={Link} to="/">Back to home</Button>
    </div>
  );
}
