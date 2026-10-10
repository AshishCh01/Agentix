import axios from "axios";
import { API_BASE_URL } from "../utils/constants";
import { supabase } from "./supabaseClient";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

let currentToken = null;

// Keep the token in sync automatically without blocking
supabase.auth.onAuthStateChange((_event, session) => {
  currentToken = session?.access_token || null;
});

// Initialize it once
supabase.auth.getSession().then(({ data }) => {
  if (data?.session?.access_token) {
    currentToken = data.session.access_token;
  }
});

apiClient.interceptors.request.use(
  async (config) => {
    // Use the cached token if available to avoid Supabase locking latency
    let token = currentToken;
    
    if (!token) {
      const { data } = await supabase.auth.getSession();
      token = data?.session?.access_token;
      currentToken = token;
    }
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const getCurrentToken = async () => {
  if (currentToken) return currentToken;
  const { data } = await supabase.auth.getSession();
  return data?.session?.access_token || null;
};

export default apiClient;