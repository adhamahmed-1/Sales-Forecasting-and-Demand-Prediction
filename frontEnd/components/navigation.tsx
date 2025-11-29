"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, Layout, MessageSquare, Users, Sparkles } from "lucide-react"

const navItems = [
  { href: "/", icon: BarChart3, label: "Predict" },
  { href: "/dashboard", icon: Layout, label: "Dashboard" },
  { href: "/chatbot", icon: MessageSquare, label: "Chat" },
  { href: "/team", icon: Users, label: "Team" },
]

export default function Navigation() {
  const pathname = usePathname()

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="relative w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 via-purple-600 to-blue-600 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Sparkles size={22} className="text-white" />
              </div>
              <span className="font-bold text-lg gradient-text">SalesForce AI</span>
            </Link>

            <div className="flex items-center gap-2">
              {navItems.map(({ href, icon: Icon, label }) => {
                const isActive = pathname === href
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                      isActive
                        ? "bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg shadow-purple-500/50"
                        : "text-muted-foreground hover:text-foreground hover:bg-gradient-to-r hover:from-purple-600/20 hover:to-blue-600/20"
                    }`}
                  >
                    <Icon size={18} />
                    <span>{label}</span>
                  </Link>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
