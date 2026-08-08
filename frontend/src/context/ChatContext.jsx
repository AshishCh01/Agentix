import React, { createContext, useState, useEffect, useCallback } from "react";
import { sessionApi } from "../api/sessionApi";
import { chatApi } from "../api/chatApi";

export const ChatContext = createContext(null);

export const ChatProvider = ({ children }) => {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeNode, setActiveNode] = useState(null);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);

  // Load user sessions on initialization
  const fetchSessions = useCallback(async () => {
    setLoadingSessions(true);
    try {
      const data = await sessionApi.getSessions();
      setSessions(data || []);
      if (data && data.length > 0 && !activeSessionId) {
        setActiveSessionId(data[0].id);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setLoadingSessions(false);
    }
  }, [activeSessionId]);

  // Fetch messages whenever active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }

    const loadMessages = async () => {
      setLoadingMessages(true);
      try {
        const history = await sessionApi.getSessionMessages(activeSessionId);
        setMessages(history || []);
      } catch (err) {
        // Handle 404 (new session with no messages yet)
        setMessages([]);
      } finally {
        setLoadingMessages(false);
      }
    };

    loadMessages();
  }, [activeSessionId]);

  // Create new session
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

  // Delete session
  const deleteSession = async (sessionId) => {
    try {
      await sessionApi.deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.id !== sessionId);
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
      throw err;
    }
  };

  // Send message and handle SSE response stream
  const sendMessage = async (userPrompt) => {
    if (!userPrompt.trim() || isStreaming) return;

    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      const created = await createNewSession(userPrompt.slice(0, 30));
      targetSessionId = created.id;
    }

    const userMsg = {
      id: Date.now().toString(),
      role: "user",
      content: userPrompt,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setActiveNode("supervisor");

    try {
      await chatApi.streamMessage({
        sessionId: targetSessionId,
        message: userPrompt,
        onEvent: (event) => {
          if (event.type === "node_start") {
            setActiveNode(event.node);
          } else if (event.type === "completion") {
            const assistantMsg = {
              id: (Date.now() + 1).toString(),
              role: "assistant",
              content: event.assistant_message,
              sources: event.sources || [],
              intent: event.intent,
              created_at: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, assistantMsg]);
            setActiveNode(null);
          }
        },
        onError: (err) => {
          console.error("Stream error:", err);
          const errorMsg = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `Error: ${err.message || "Failed to stream response. Please re-authenticate."}`,
            created_at: new Date().toISOString(),
          };
          setMessages((prev) => [...prev, errorMsg]);
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