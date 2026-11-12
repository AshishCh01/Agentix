import { lazy, Suspense } from "react";
import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { FullPageLoader } from "../components/Common/Loader";
import { ProtectedRoute, PublicOnlyRoute } from "../components/Common/RouteGuards";

// Route-level code splitting keeps the landing page small; the markdown/KaTeX
// bundle only loads once someone opens the workspace.
const Landing = lazy(() => import("../pages/Landing"));
const Login = lazy(() => import("../pages/Login"));
const Register = lazy(() => import("../pages/Register"));
const NotFound = lazy(() => import("../pages/NotFound"));
const AuthLayout = lazy(() => import("../layouts/AuthLayout"));
const WorkspaceProviders = lazy(() => import("../layouts/WorkspaceProviders"));
const AppLayout = lazy(() => import("../layouts/AppLayout"));
const ChatPage = lazy(() => import("../pages/ChatPage"));
const KnowledgeBaseListPage = lazy(() => import("../pages/KnowledgeBaseListPage"));
const KnowledgeBaseDetailPage = lazy(() => import("../pages/KnowledgeBaseDetailPage"));

const LegacyKbDetail = () => {
  const { id } = useParams();
  return <Navigate to={`/app/knowledge-bases/${id}`} replace />;
};

export const AppRoutes = () => (
  <Suspense fallback={<FullPageLoader />}>
    <Routes>
      <Route path="/" element={<Landing />} />

      <Route element={<PublicOnlyRoute />}>
        <Route element={<AuthLayout />}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
        </Route>
      </Route>

      <Route element={<ProtectedRoute />}>
        <Route element={<WorkspaceProviders />}>
          <Route path="/app" element={<AppLayout />}>
            <Route index element={<ChatPage />} />
            <Route path="knowledge-bases" element={<KnowledgeBaseListPage />} />
            <Route path="knowledge-bases/:id" element={<KnowledgeBaseDetailPage />} />
          </Route>
        </Route>
      </Route>

      {/* Old URLs from before the landing page existed */}
      <Route path="/knowledge-bases" element={<Navigate to="/app/knowledge-bases" replace />} />
      <Route path="/knowledge-bases/:id" element={<LegacyKbDetail />} />

      <Route path="*" element={<NotFound />} />
    </Routes>
  </Suspense>
);
