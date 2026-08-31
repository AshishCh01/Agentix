import { useContext } from "react";
import { KnowledgeBaseContext } from "../context/KnowledgeBaseContext";

export const useKnowledgeBase = () => {
  const context = useContext(KnowledgeBaseContext);
  if (!context) {
    throw new Error("useKnowledgeBase must be used within a KnowledgeBaseProvider");
  }
  return context;
};
