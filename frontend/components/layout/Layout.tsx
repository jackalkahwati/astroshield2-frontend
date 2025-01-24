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
    <div className="relative min-h-screen">
      <div className="hidden md:flex">
        <Sidebar />
      </div>
      <main className={cn(
        "flex min-h-screen flex-col transition-all duration-300",
        isOpen ? "md:pl-64" : "md:pl-[72px]"
      )}>
        <Header />
        <div className="flex-1 space-y-4 p-8 pt-6">
          {children}
        </div>
      </main>
    </div>
  )
} 