import { Outlet } from "react-router-dom";
import { KnowledgeBaseProvider } from "../context/KnowledgeBaseContext";
import { ChatProvider } from "../context/ChatContext";

/** Data providers only mount for signed-in workspace routes, never on the landing page. */
export default function WorkspaceProviders() {
  return (
    <KnowledgeBaseProvider>
      <ChatProvider>
        <Outlet />
      </ChatProvider>
    </KnowledgeBaseProvider>
  );
}
