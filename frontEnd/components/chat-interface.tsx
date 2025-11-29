"use client"

import { useState, useRef, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Sparkles } from "lucide-react"

interface Message {
  id: string
  text: string
  sender: "user" | "bot"
  timestamp: Date
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! I'm your AI sales prediction assistant. How can I help you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
  if (!input.trim()) return;

  const userMessage: Message = {
    id: Date.now().toString(),
    text: input,
    sender: "user",
    timestamp: new Date(),
  };

  setMessages((prev) => [...prev, userMessage]);
  setInput("");
  setIsLoading(true);

  try {
    // SEND MESSAGE TO FASTAPI
    const res = await fetch("http://localhost:8001/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage.text }),
    });

    const data = await res.json();

    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: data.reply, // reply from Gemini
      sender: "bot",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, botMessage]);
  } catch (err) {
    console.error(err);

    const errorMessage: Message = {
      id: (Date.now() + 2).toString(),
      text: "⚠️ Error: Could not reach AI server.",
      sender: "bot",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, errorMessage]);
  }

  setIsLoading(false);
};


  return (
    <Card className="flex flex-col h-96 md:h-[500px] bg-gradient-to-br from-slate-800 to-slate-900 border border-purple-500/30 card-glow fade-in-scale">
      {/* Chat Header */}
      <div className="border-b border-purple-500/30 px-6 py-4 bg-gradient-to-r from-purple-600/10 to-blue-600/10">
        <div className="flex items-center gap-2">
          <Sparkles size={20} className="text-purple-400" />
          <h2 className="text-lg font-semibold gradient-text">Sales AI Chatbot</h2>
        </div>
        <p className="text-xs text-muted-foreground mt-1">Powered by advanced ML models</p>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"} fade-in-scale`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg transition-all duration-300 ${
                message.sender === "user"
                  ? "bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-br-none shadow-lg shadow-purple-500/30"
                  : "bg-gradient-to-r from-slate-700 to-slate-800 text-foreground rounded-bl-none border border-purple-500/20"
              }`}
            >
              <p className="text-sm leading-relaxed">{message.text}</p>
              <p className={`text-xs mt-2 ${message.sender === "user" ? "text-white/70" : "text-muted-foreground"}`}>
                {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start fade-in-scale">
            <div className="bg-gradient-to-r from-slate-700 to-slate-800 text-foreground px-4 py-3 rounded-lg rounded-bl-none border border-purple-500/20">
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                  style={{ animationDelay: "100ms" }}
                ></div>
                <div
                  className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                  style={{ animationDelay: "200ms" }}
                ></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-purple-500/30 px-6 py-4 bg-gradient-to-r from-purple-600/5 to-blue-600/5">
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="Ask about your predictions..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            className="flex-1 bg-slate-800 text-foreground border-purple-500/30 focus:ring-purple-500 focus:border-transparent"
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg shadow-purple-500/50"
          >
            <Send size={18} />
          </Button>
        </div>
      </div>
    </Card>
  )
}
