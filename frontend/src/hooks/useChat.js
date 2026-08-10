import { useContext, useRef, useEffect } from "react";
import { ChatContext } from "../context/ChatContext";

// Primary hook consumed by UploadButton, MessageList, ChatContainer, etc.
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};

// Streaming hook with AbortController for stream cancellation
export const useChatStream = () => {
  const abortControllerRef = useRef(null);

  const sendMessageStream = async (sessionId, query, onToken) => {
    // Cancel any active stream before starting a new request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({ session_id: sessionId, user_query: query }),
        signal: abortControllerRef.current.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        onToken(chunk);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("Stream request canceled by user navigation.");
      } else {
        console.error("Streaming error:", err);
      }
    }
  };

  useEffect(() => {
    return () => {
      // Abort active streams when component unmounts
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return { sendMessageStream };
};