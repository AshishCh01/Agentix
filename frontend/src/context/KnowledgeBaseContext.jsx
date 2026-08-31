/* eslint-disable react-refresh/only-export-components */
import { createContext, useState, useEffect, useCallback, useContext } from "react";
import { knowledgeBaseApi } from "../api/knowledgeBaseApi";
import { AuthContext } from "./AuthContext";
import { getErrorMessage } from "../utils/helpers";

export const KnowledgeBaseContext = createContext(null);

export const KnowledgeBaseProvider = ({ children }) => {
  const { user } = useContext(AuthContext);

  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchKnowledgeBases = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await knowledgeBaseApi.getKnowledgeBases();
      setKnowledgeBases(data || []);
    } catch (err) {
      setError(getErrorMessage(err));
      setKnowledgeBases([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!user) {
      const clearTimeoutId = setTimeout(() => setKnowledgeBases([]), 0);
      return () => clearTimeout(clearTimeoutId);
    }

    (async () => {
      await fetchKnowledgeBases();
    })();
  }, [user, fetchKnowledgeBases]);

  const createKnowledgeBase = async (name, description) => {
    const kb = await knowledgeBaseApi.createKnowledgeBase(name, description);
    setKnowledgeBases((prev) => [kb, ...prev]);
    return kb;
  };

  const updateKnowledgeBase = async (kbId, fields) => {
    const updated = await knowledgeBaseApi.updateKnowledgeBase(kbId, fields);
    setKnowledgeBases((prev) => prev.map((kb) => (kb.id === kbId ? updated : kb)));
    return updated;
  };

  const deleteKnowledgeBase = async (kbId) => {
    await knowledgeBaseApi.deleteKnowledgeBase(kbId);
    setKnowledgeBases((prev) => prev.filter((kb) => kb.id !== kbId));
  };

  return (
    <KnowledgeBaseContext.Provider
      value={{
        knowledgeBases,
        loading,
        error,
        fetchKnowledgeBases,
        createKnowledgeBase,
        updateKnowledgeBase,
        deleteKnowledgeBase,
      }}
    >
      {children}
    </KnowledgeBaseContext.Provider>
  );
};
