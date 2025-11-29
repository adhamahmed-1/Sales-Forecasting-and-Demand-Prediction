import type React from "react"
import Navigation from "./navigation"

export default function LayoutWrapper({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
    </div>
  )
}
