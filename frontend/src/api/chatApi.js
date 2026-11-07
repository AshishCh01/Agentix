import apiClient, { getCurrentToken } from "./axios";
import { API_BASE_URL } from "../utils/constants";
import { parseSSEStream } from "../utils/streamParser";
import { supabase } from "./supabaseClient";

export const chatApi = {
  sendMessageSync: async (sessionId, message, imageData = null, allowWebSearch = false) => {
    const response = await apiClient.post("/chat", {
      session_id: sessionId,
      message,
      image_data: imageData,
      allow_web_search: Boolean(allowWebSearch),
    });
    return response.data;
  },

  streamMessage: async ({
    sessionId,
    message,
    imageData = null,
    allowWebSearch = false,
    allow_web_search,
    onEvent,
    onError,
    signal,
  }) => {
    try {
      const enableWebSearch = Boolean(allowWebSearch || allow_web_search);
      let token = await getCurrentToken();

      if (!token) {
        const { data: refreshed } = await supabase.auth.refreshSession();
        token = refreshed?.session?.access_token;
      }

      if (!token) {
        throw new Error("Authentication token expired. Please log in again.");
      }

      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          message,
          image_data: imageData,
          allow_web_search: enableWebSearch,
        }),
        signal,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      for await (const eventPayload of parseSSEStream(reader)) {
        if (onEvent) onEvent(eventPayload);
      }
    } catch (err) {
      if (onError) onError(err);
      throw err;
    }
  },
};