import apiClient from "./axios";
import { API_BASE_URL } from "../utils/constants";
import { parseSSEStream } from "../utils/streamParser";
import { supabase } from "./supabaseClient";

export const chatApi = {
  sendMessageSync: async (sessionId, message, imageData = null) => {
    const response = await apiClient.post("/chat", {
      session_id: sessionId,
      message,
      image_data: imageData,
    });
    return response.data;
  },

  streamMessage: async ({ sessionId, message, imageData = null, onEvent, onError }) => {
    try {
      let { data } = await supabase.auth.getSession();
      let token = data?.session?.access_token;

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
        }),
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