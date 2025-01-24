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
    <div className="min-h-screen">
      <Sidebar />
      <div className={cn(
        "flex min-h-screen flex-col transition-all",
        isOpen ? "lg:pl-64" : "lg:pl-[72px]"
      )}>
        <Header />
        <main className="flex-1">
          <div className="container mx-auto p-6 pt-4">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
} 