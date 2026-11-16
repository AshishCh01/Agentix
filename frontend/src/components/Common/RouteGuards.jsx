import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { FullPageLoader } from "./Loader";

/** Workspace routes: signed-in users only. Remembers where they were headed. */
export const ProtectedRoute = () => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();
  if (loading) return <FullPageLoader />;
  if (!isAuthenticated) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <Outlet />;
};

/** Login / register: bounce signed-in users straight to the workspace. */
export const PublicOnlyRoute = () => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();
  if (loading) return <FullPageLoader />;
  if (isAuthenticated) return <Navigate to={location.state?.from || "/app"} replace />;
  return <Outlet />;
};
