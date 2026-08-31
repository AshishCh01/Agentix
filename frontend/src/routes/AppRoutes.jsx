import { Routes, Route, Navigate } from "react-router-dom";
import { ProtectedRoute } from "../components/Common/ProtectedRoute";
import { AuthLayout } from "../layouts/AuthLayout";
import { ChatLayout } from "../layouts/ChatLayout";
import { Login } from "../pages/Login";
import { Register } from "../pages/Register";
import { ChatPage } from "../pages/ChatPage";
import { KnowledgeBaseListPage } from "../pages/KnowledgeBaseListPage";
import { KnowledgeBaseDetailPage } from "../pages/KnowledgeBaseDetailPage";
import { NotFound } from "../pages/NotFound";

export const AppRoutes = () => {
  return (
    <Routes>
      {/* Public Auth Routes */}
      <Route element={<AuthLayout />}>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
      </Route>

      {/* Protected Workspace Routes */}
      <Route element={<ProtectedRoute />}>
        <Route element={<ChatLayout />}>
          <Route path="/" element={<ChatPage />} />
          <Route path="/knowledge-bases" element={<KnowledgeBaseListPage />} />
          <Route path="/knowledge-bases/:id" element={<KnowledgeBaseDetailPage />} />
        </Route>
      </Route>

      {/* Fallback */}
      <Route path="404" element={<NotFound />} />
      <Route path="*" element={<Navigate to="/404" replace />} />
    </Routes>
  );
};