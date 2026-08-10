import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles, ImagePlus, X } from "lucide-react";
import { useChat } from "../../hooks/useChat";

export const ChatInput = () => {
  const [prompt, setPrompt] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const { sendMessage, isStreaming } = useChat();
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        160
      )}px`;
    }
  }, [prompt]);

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setSelectedImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = (e) => {
    e?.preventDefault();
    if ((!prompt.trim() && !selectedImage) || isStreaming) return;

    const textToSend = prompt;
    const imageToSend = selectedImage;

    setPrompt("");
    removeImage();

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    sendMessage(textToSend, imageToSend);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 bg-slate-900/90 border-t border-slate-800 backdrop-blur sticky bottom-0 z-10">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative">
        {selectedImage && (
          <div className="mb-2 relative inline-block group">
            <img
              src={selectedImage}
              alt="Selected attachment preview"
              className="w-16 h-16 object-cover rounded-xl border border-slate-700 shadow-lg"
            />
            <button
              type="button"
              onClick={removeImage}
              className="absolute -top-1.5 -right-1.5 bg-slate-800 border border-slate-600 text-slate-300 hover:bg-rose-600 hover:text-white p-0.5 rounded-full transition-all shadow-md"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        <div className="relative flex items-center bg-slate-950 border border-slate-800 rounded-2xl focus-within:border-blue-500/80 focus-within:ring-1 focus-within:ring-blue-500/50 transition-all shadow-xl">
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isStreaming}
            className="pl-3.5 pr-1 text-slate-400 hover:text-blue-400 disabled:opacity-50 transition-colors flex items-center justify-center"
            title="Attach image"
          >
            <ImagePlus className="w-5 h-5" />
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question or attach an image..."
            disabled={isStreaming}
            className="w-full py-3.5 pl-2 pr-12 bg-transparent text-sm text-slate-100 placeholder-slate-500 resize-none focus:outline-none max-h-40 custom-scrollbar"
          />

          <button
            type="submit"
            disabled={(!prompt.trim() && !selectedImage) || isStreaming}
            className="absolute right-2.5 p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-xl transition-all flex items-center justify-center flex-shrink-0"
          >
            {isStreaming ? (
              <Sparkles className="w-4 h-4 animate-spin text-blue-400" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-500 px-2 mt-2">
          <span>Press Enter to send, Shift + Enter for new line</span>
          <span>Powered by Agentic RAG Pipeline</span>
        </div>
      </form>
    </div>
  );
};