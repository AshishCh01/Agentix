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
        <div className="p-4 my-2 border border-red-200 bg-red-50 text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400 rounded-lg text-sm">
          <p className="font-semibold">Failed to render this message stream.</p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="mt-2 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 dark:bg-red-500/20 dark:hover:bg-red-500/30 dark:text-red-200 rounded text-xs transition"
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