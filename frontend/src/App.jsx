import { BrowserRouter } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import { AuthProvider } from "./context/AuthContext";
import { ChatProvider } from "./context/ChatContext";
import { KnowledgeBaseProvider } from "./context/KnowledgeBaseContext";
import { AppRoutes } from "./routes/AppRoutes";

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <KnowledgeBaseProvider>
            <ChatProvider>
              <AppRoutes />
            </ChatProvider>
          </KnowledgeBaseProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}