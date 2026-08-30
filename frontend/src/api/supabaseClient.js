import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error("Missing Supabase environment variables in .env file");
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    // Tab-scoped storage instead of the default localStorage: the session
    // is gone as soon as the tab/browser closes and isn't shared across
    // tabs, shrinking how long a stolen token stays valid and how far it
    // can leak if an XSS bug ever reads browser storage.
    storage: window.sessionStorage,
    storageKey: "agentic-rag-auth-token",
    persistSession: true,
    autoRefreshToken: true,
  },
});