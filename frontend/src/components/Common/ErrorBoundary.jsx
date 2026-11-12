import { Component } from "react";
import { TriangleAlert } from "lucide-react";
import { StateMessage } from "../ui/StateMessage";
import { Button } from "../ui/Button";

class ErrorBoundary extends Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("React error boundary caught:", error, info);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    if (this.props.fullPage) {
      return (
        <div className="flex min-h-dvh items-center justify-center bg-bg">
          <StateMessage
            icon={TriangleAlert}
            tone="danger"
            title="Something went wrong"
            description="The page hit an unexpected error. Reloading usually fixes it."
            action={<Button onClick={() => window.location.reload()}>Reload page</Button>}
          />
        </div>
      );
    }

    return (
      <div role="alert" className="my-2 rounded-lg border border-danger/30 bg-danger-soft p-3 text-sm text-danger">
        <p className="font-medium">This message couldn’t be displayed.</p>
        <button
          type="button"
          onClick={() => this.setState({ hasError: false })}
          className="mt-1 text-xs underline underline-offset-2"
        >
          Try again
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
