"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"

const SidebarContext = createContext<{
  isCollapsed: boolean;
  toggleSidebar: () => void;
} | null>(null)

function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const toggleSidebar = () => setIsCollapsed(!isCollapsed)

  return <SidebarContext.Provider value={{ isCollapsed, toggleSidebar }}>{children}</SidebarContext.Provider>
}

function useSidebar() {
  const context = useContext(SidebarContext)
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider")
  }
  return context
}

export { SidebarProvider, useSidebar }

