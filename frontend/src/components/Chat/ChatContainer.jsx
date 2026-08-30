import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";

export const ChatContainer = () => {
  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col bg-white dark:bg-slate-900">
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <MessageList />
      </div>
      <ChatInput />
    </div>
  );
};