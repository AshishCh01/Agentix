import { createContext, useState, useEffect, useCallback, useContext } from "react";
import { sessionApi } from "../api/sessionApi";
import { chatApi } from "../api/chatApi";
import { AuthContext } from "./AuthContext";

export const ChatContext = createContext(null);

export const ChatProvider = ({ children }) => {
  const { user } = useContext(AuthContext);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeNode, setActiveNode] = useState(null);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);

  const fetchSessions = useCallback(async () => {
    setLoadingSessions(true);
    try {
      const data = await sessionApi.getSessions();
      const userSessions = data || [];
      setSessions(userSessions);
      if (userSessions.length > 0) {
        setActiveSessionId(userSessions[0].id);
      } else {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
      setSessions([]);
      setActiveSessionId(null);
      setMessages([]);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    if (!user) {
      setSessions([]);
      setMessages([]);
      setActiveSessionId(null);
      return;
    }

    const initUserData = async () => {
      setLoadingSessions(true);
      try {
        const data = await sessionApi.getSessions();
        if (!isMounted) return;
        const userSessions = data || [];
        setSessions(userSessions);
        if (userSessions.length > 0) {
          setActiveSessionId(userSessions[0].id);
        } else {
          setActiveSessionId(null);
          setMessages([]);
        }
      } catch (err) {
        if (!isMounted) return;
        console.error("Failed to load user sessions:", err);
        setSessions([]);
        setActiveSessionId(null);
        setMessages([]);
      } finally {
        if (isMounted) setLoadingSessions(false);
      }
    };

    initUserData();

    return () => {
      isMounted = false;
    };
  }, [user]);

  useEffect(() => {
    if (!activeSessionId) return;

    let isMounted = true;

    const loadMessages = async () => {
      setLoadingMessages(true);
      try {
        const history = await sessionApi.getSessionMessages(activeSessionId);
        if (isMounted) {
          setMessages(history || []);
        }
      } catch {
        if (isMounted) {
          setMessages([]);
        }
      } finally {
        if (isMounted) {
          setLoadingMessages(false);
        }
      }
    };

    loadMessages();

    return () => {
      isMounted = false;
    };
  }, [activeSessionId]);

  const createNewSession = async (title = "New Chat Session") => {
    try {
      const newSession = await sessionApi.createSession(title);
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      setMessages([]);
      return newSession;
    } catch (err) {
      console.error("Failed to create session:", err);
      throw err;
    }
  };

  const deleteSession = async (sessionId) => {
    try {
      await sessionApi.deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.id !== sessionId);
        const nextId = remaining.length > 0 ? remaining[0].id : null;
        setActiveSessionId(nextId);
        if (!nextId) setMessages([]);
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
      throw err;
    }
  };

  const sendMessage = async (userPrompt, imageData = null) => {
    if ((!userPrompt.trim() && !imageData) || isStreaming) return;

    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      const sessionTitle = userPrompt.trim()
        ? userPrompt.slice(0, 30)
        : "Image Query";
      const created = await createNewSession(sessionTitle);
      targetSessionId = created.id;
    }

    const userMsg = {
      id: Date.now().toString(),
      role: "user",
      content: userPrompt,
      image_data: imageData,
      created_at: new Date().toISOString(),
    };

    const assistantPlaceholder = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      sources: [],
      isStreaming: true,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setIsStreaming(true);
    setActiveNode("supervisor");

    try {
      await chatApi.streamMessage({
        sessionId: targetSessionId,
        message: userPrompt,
        imageData,
        onEvent: (event) => {
          if (event.type === "node_start") {
            setActiveNode(event.node);
          } else if (event.type === "token") {
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              if (lastMsg && lastMsg.role === "assistant") {
                return [
                  ...updated.slice(0, -1),
                  {
                    ...lastMsg,
                    content: (lastMsg.content || "") + (event.content || ""),
                  },
                ];
              }
              return updated;
            });
          } else if (event.type === "completion") {
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              if (lastMsg && lastMsg.role === "assistant") {
                return [
                  ...updated.slice(0, -1),
                  {
                    ...lastMsg,
                    content: event.assistant_message || lastMsg.content,
                    sources: event.sources || [],
                    intent: event.intent,
                    reflection: event.reflection || null,
                    isStreaming: false,
                  },
                ];
              }
              return updated;
            });
            setActiveNode(null);
          }
        },
        onError: (err) => {
          console.error("Stream error:", err);
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              return [
                ...updated.slice(0, -1),
                {
                  ...lastMsg,
                  content: `Error: ${err.message || "Failed to stream response. Please re-authenticate."}`,
                  isStreaming: false,
                },
              ];
            }
            return updated;
          });
          setActiveNode(null);
        },
      });
    } catch (err) {
      console.error("Failed to stream response:", err);
    } finally {
      setIsStreaming(false);
      setActiveNode(null);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        sessions,
        activeSessionId,
        setActiveSessionId,
        messages,
        isStreaming,
        activeNode,
        loadingSessions,
        loadingMessages,
        fetchSessions,
        createNewSession,
        deleteSession,
        sendMessage,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};