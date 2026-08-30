/* eslint-disable react-refresh/only-export-components */
import { createContext, useState, useEffect, useCallback, useContext, useRef } from "react";
import { sessionApi } from "../api/sessionApi";
import { chatApi } from "../api/chatApi";
import { AuthContext } from "./AuthContext";
import { MAX_RETAINED_MESSAGE_IMAGES } from "../utils/constants";

export const ChatContext = createContext(null);

// Keeps only the most recent MAX_RETAINED_MESSAGE_IMAGES messages' base64
// image_data in memory, stripping it from older ones. Without this, a long
// chat session with many image attachments would accumulate unbounded
// base64 payloads in React state for the lifetime of the tab.
const pruneOldImageData = (msgs) => {
  let seen = 0;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (!msgs[i].image_data) continue;
    seen += 1;
    if (seen > MAX_RETAINED_MESSAGE_IMAGES) {
      msgs = [
        ...msgs.slice(0, i),
        { ...msgs[i], image_data: null },
        ...msgs.slice(i + 1),
      ];
    }
  }
  return msgs;
};

export const ChatProvider = ({ children }) => {
  const { user } = useContext(AuthContext);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeNode, setActiveNode] = useState(null);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);

  const skipNextFetch = useRef(null);

  // Tracks the in-flight SSE stream so its effects can be isolated from
  // whatever session is actually active by the time each event arrives, and
  // so it can be cancelled outright on session switch, deletion, logout, or
  // unmount instead of silently writing into the wrong session's messages.
  const streamControllerRef = useRef(null);
  const streamSessionIdRef = useRef(null);
  const activeSessionIdRef = useRef(activeSessionId);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  // Aborts the underlying fetch only — no state updates, safe to call from
  // an unmount cleanup.
  const abortController = useCallback(() => {
    if (streamControllerRef.current) {
      streamControllerRef.current.abort();
      streamControllerRef.current = null;
    }
    streamSessionIdRef.current = null;
  }, []);

  // Aborts the stream and resets the streaming UI state. Use this from any
  // live interaction path (session switch, deletion, logout) — never from
  // an unmount cleanup, where setting state is unsafe.
  const abortActiveStream = useCallback(() => {
    abortController();
    setIsStreaming(false);
    setActiveNode(null);
  }, [abortController]);

  // Defense in depth: if the active session ever changes out from under an
  // in-flight stream through a path other than sendMessage's own bookkeeping
  // (e.g. a future navigation feature), abandon that stream immediately.
  useEffect(() => {
    if (streamSessionIdRef.current && streamSessionIdRef.current !== activeSessionId) {
      setTimeout(() => abortActiveStream(), 0);
    }
  }, [activeSessionId, abortActiveStream]);

  // Cancel any in-flight stream if the provider itself unmounts.
  useEffect(() => {
    return () => {
      abortController();
    };
  }, [abortController]);

  // Guards against a stale request clobbering state: only the response to
  // the most recently issued fetchSessions() call is allowed to commit.
  // Shared by every caller (the login effect below and any manual refresh)
  // so this race protection lives in exactly one place.
  const fetchRequestIdRef = useRef(0);

  const fetchSessions = useCallback(async () => {
    const requestId = ++fetchRequestIdRef.current;
    setLoadingSessions(true);
    try {
      const data = await sessionApi.getSessions();
      if (fetchRequestIdRef.current !== requestId) return;
      const userSessions = data || [];
      setSessions(userSessions);
      if (userSessions.length > 0) {
        setActiveSessionId(userSessions[0].id);
      } else {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (err) {
      if (fetchRequestIdRef.current !== requestId) return;
      console.error("Failed to load sessions:", err);
      setSessions([]);
      setActiveSessionId(null);
      setMessages([]);
    } finally {
      if (fetchRequestIdRef.current === requestId) setLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    if (!user) {
      const clearTimeoutId = setTimeout(() => {
        abortActiveStream();
        setSessions([]);
        setMessages([]);
        setActiveSessionId(null);
      }, 0);
      // If the user logs back in before this fires (React re-runs this
      // effect with the new `user`), cancel it -- otherwise this stale
      // clear can land after the re-login's fetchSessions() has already
      // populated state, wiping the newly logged-in user's data.
      return () => clearTimeout(clearTimeoutId);
    }

    // Wrapped so the effect body itself never synchronously calls setState
    // (fetchSessions does, to flip on the loading flag) -- only kicks off
    // the fetch, deferred to a microtask.
    (async () => {
      await fetchSessions();
    })();
  }, [user, abortActiveStream, fetchSessions]);

  useEffect(() => {
    if (!activeSessionId) return;

    let isMounted = true;

    if (skipNextFetch.current === activeSessionId) {
      skipNextFetch.current = null;
      return;
    }

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
      skipNextFetch.current = newSession.id;
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
      if (streamSessionIdRef.current === sessionId) {
        abortActiveStream();
      }
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
      // Set synchronously rather than waiting on the activeSessionId sync
      // effect to commit — otherwise a fast SSE event for a brand-new
      // session could arrive before that effect runs and get dropped by
      // isStillActive() below.
      activeSessionIdRef.current = targetSessionId;
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

    setMessages((prev) => pruneOldImageData([...prev, userMsg, assistantPlaceholder]));
    setIsStreaming(true);
    setActiveNode("supervisor");

    // Cancel any stray previous stream (should already be idle given the
    // isStreaming guard above, but this keeps a superseded stream from ever
    // writing into this new one's messages) and start tracking this one.
    abortController();
    const controller = new AbortController();
    streamControllerRef.current = controller;
    streamSessionIdRef.current = targetSessionId;

    // Only true while the user is still looking at the session this stream
    // was started for — false the moment they switch away, delete it, or
    // log out. Every state update below is gated on this so a late-arriving
    // token/completion/error for an abandoned stream can never land in a
    // different session's message list.
    const isStillActive = () => activeSessionIdRef.current === targetSessionId;

    try {
      await chatApi.streamMessage({
        sessionId: targetSessionId,
        message: userPrompt,
        imageData,
        signal: controller.signal,
        onEvent: (event) => {
          if (!isStillActive()) return;
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
          if (err?.name === "AbortError") {
            // Intentional cancellation (session switch/deletion/logout/
            // unmount) — not a real error, nothing to surface.
            return;
          }
          if (!isStillActive()) return;
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
      if (err?.name !== "AbortError") {
        console.error("Failed to stream response:", err);
      }
    } finally {
      // Only clear state if nothing has superseded this stream (e.g. a new
      // sendMessage call already replaced streamControllerRef with its own
      // controller) — otherwise this stale finally would clobber the newer
      // stream's in-progress state.
      if (streamControllerRef.current === controller) {
        streamControllerRef.current = null;
        streamSessionIdRef.current = null;
        setIsStreaming(false);
        setActiveNode(null);
      }
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
        cancelStream: abortActiveStream,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};