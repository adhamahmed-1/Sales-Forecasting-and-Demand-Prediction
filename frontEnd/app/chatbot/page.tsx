"use client"

import LayoutWrapper from "@/components/layout-wrapper"
import ChatInterface from "@/components/chat-interface"

export default function ChatbotPage() {
  return (
    <LayoutWrapper>
      <div className="space-y-6">
        <div>
          <h1 className="text-5xl font-bold mb-2 gradient-text">AI Assistant</h1>
          <p className="text-lg text-muted-foreground">Get insights about your sales predictions and data</p>
        </div>
        <ChatInterface />
      </div>
    </LayoutWrapper>
  )
}
