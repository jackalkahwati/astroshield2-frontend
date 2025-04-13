"use client"

import { ReactNode } from "react"
import { TopBar } from "@/components/layout/top-bar"
import { Sidebar } from "@/components/layout/sidebar"

interface ShellProps {
  children: ReactNode
}

export function Shell({ children }: ShellProps) {
  return (
    <div className="relative flex min-h-screen">
      <Sidebar />
      <div className="flex-1">
        <TopBar />
        <main className="flex-1 space-y-4 p-8 pt-6">
          {children}
        </main>
      </div>
    </div>
  )
} 