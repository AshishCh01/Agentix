import React, { useState } from "react";
import { Plus } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { Spinner } from "../Common/Loader";

export const NewChatButton = () => {
  const { createNewSession, isStreaming } = useChat();
  const [loading, setLoading] = useState(false);

  const handleCreateNewChat = async () => {
    if (isStreaming || loading) return;
    setLoading(true);
    try {
      await createNewSession("New Chat Session");
    } catch (err) {
      console.error("Error creating new chat:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleCreateNewChat}
      disabled={isStreaming || loading}
      className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition-colors shadow-sm"
    >
      {loading ? (
        <Spinner size="sm" className="text-white" />
      ) : (
        <>
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </>
      )}
    </button>
  );
};