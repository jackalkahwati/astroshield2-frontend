"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "./sidebar"
import { TopBar } from "./top-bar"
import { cn } from "@/lib/utils"
import { useToast } from "@/components/ui/use-toast"
import { Loader2 } from "lucide-react"
import { SidebarProvider } from "@/components/ui/sidebar"
import { FeedbackWidget } from "@/components/feedback-support/feedback-widget"

interface RootLayoutProps {
  children: React.ReactNode
}

export function RootLayout({ children }: RootLayoutProps) {
  const [isMounted, setIsMounted] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    setIsMounted(true)
    // Simulate initial data loading
    setTimeout(() => {
      setIsLoading(false)
    }, 1000)
  }, [])

  useEffect(() => {
    // System status notification
    toast({
      title: "System Status",
      description: "All systems operational",
      duration: 3000,
    })
  }, [toast])

  if (!isMounted) {
    return null
  }

  return (
    <SidebarProvider>
      <div className="min-h-screen">
        <div className="flex">
          <Sidebar className="hidden md:flex" />
          <div className="flex-1 flex flex-col min-h-screen">
            <TopBar />
            <div className="flex-1 overflow-auto">
              {isLoading ? (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : (
                <div className="container mx-auto p-6">{children}</div>
              )}
            </div>
          </div>
        </div>
        <FeedbackWidget />
      </div>
    </SidebarProvider>
  )
}

