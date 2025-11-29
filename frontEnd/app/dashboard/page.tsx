"use client"
import LayoutWrapper from "@/components/layout-wrapper"
import Dashboard from "@/components/dashboard"

export default function DashboardPage() {
  return (
    <LayoutWrapper>
      <div className="space-y-6">
        <div>
          <h1 className="text-5xl font-bold mb-2 gradient-text">Analytics Dashboard</h1>
          <p className="text-lg text-muted-foreground">Upload data and explore your sales metrics with AI insights</p>
        </div>
        <Dashboard />
      </div>
    </LayoutWrapper>
  )
}
