import { useEffect } from "react";

export const usePageTitle = (title) => {
  useEffect(() => {
    document.title = title ? `${title} · Agentix` : "Agentix — Ask your documents. Get answers you can check.";
  }, [title]);
};
