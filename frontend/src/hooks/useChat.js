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

// Streaming hook with AbortController and SSE buffer handling
export const useChatStream = () => {
  const abortControllerRef = useRef(null);

  const sendMessageStream = async (sessionId, query, onToken) => {
    // Cancel active stream before starting a new request
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
        // Updated field name to 'message' to match ChatRequest schema
        body: JSON.stringify({ session_id: sessionId, message: query }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep incomplete tail in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            const rawJson = trimmed.replace(/^data:\s*/, "");
            try {
              const parsed = JSON.parse(rawJson);
              onToken(parsed);
            } catch (e) {
              // Plain string token fallback
              onToken(rawJson);
            }
          }
        }
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
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return { sendMessageStream };
};