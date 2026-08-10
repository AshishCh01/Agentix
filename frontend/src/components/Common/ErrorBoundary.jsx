import { Component } from "react";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("React Error Boundary Caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 my-2 border border-red-500/30 bg-red-500/10 rounded-lg text-red-400 text-sm">
          <p className="font-semibold">Failed to render this message stream.</p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="mt-2 px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-200 rounded text-xs transition"
          >
            Retry Rendering
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;