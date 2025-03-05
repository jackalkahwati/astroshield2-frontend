"use client"

import { createContext, useState, useContext, ReactNode, useEffect } from "react"

interface SidebarContextValue {
  isOpen: boolean
  toggle: () => void
  setIsOpen: (value: boolean) => void
}

const SidebarContext = createContext<SidebarContextValue | undefined>(undefined)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(true)
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  const toggle = () => setIsOpen((prev) => !prev)

  // Only render children when mounted to avoid hydration issues
  if (!isMounted) {
    return null
  }

  return (
    <SidebarContext.Provider value={{ isOpen, toggle, setIsOpen }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const ctx = useContext(SidebarContext)
  if (!ctx) {
    throw new Error("useSidebar must be used within SidebarProvider")
  }
  return ctx
} 