import { useContext } from "react";
import { ChatContext } from "../context/ChatContext";

// Primary hook consumed by UploadButton, MessageList, ChatContainer, etc.
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};