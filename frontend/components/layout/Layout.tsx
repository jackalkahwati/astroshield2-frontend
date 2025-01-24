"use client"

import { cn } from "@/lib/utils"
import { useSidebar } from "@/components/providers/sidebar-provider"
import { Sidebar } from "./sidebar"
import { Header } from "./header"

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const { isOpen } = useSidebar()

  return (
    <div className="relative flex h-screen overflow-hidden">
      <Sidebar />
      <main className={cn(
        "flex-1 overflow-y-auto transition-all",
        isOpen ? "lg:pl-64" : "lg:pl-20"
      )}>
        <Header />
        <div className="container mx-auto p-6 space-y-6">
          {children}
        </div>
      </main>
    </div>
  )
} 