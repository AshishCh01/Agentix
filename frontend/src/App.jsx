import { BrowserRouter } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import { AuthProvider } from "./context/AuthContext";
import { UIProvider } from "./context/UIContext";
import ErrorBoundary from "./components/Common/ErrorBoundary";
import { AppRoutes } from "./routes/AppRoutes";

export default function App() {
  return (
    <ThemeProvider>
      <UIProvider>
        <BrowserRouter>
          <AuthProvider>
            <ErrorBoundary fullPage>
              <AppRoutes />
            </ErrorBoundary>
          </AuthProvider>
        </BrowserRouter>
      </UIProvider>
    </ThemeProvider>
  );
}
