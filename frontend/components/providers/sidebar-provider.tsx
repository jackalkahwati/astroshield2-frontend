"use client"

import * as React from "react"
import { createContext, useContext, useState } from "react"

interface SidebarContextType {
  isOpen: boolean
  toggle: () => void
}

const SidebarContext = createContext<SidebarContextType>({
  isOpen: true,
  toggle: () => {},
})

interface SidebarProviderProps {
  children: React.ReactNode
}

export function SidebarProvider({
  children,
}: SidebarProviderProps) {
  const [isOpen, setIsOpen] = useState(true)

  const toggle = () => {
    setIsOpen(!isOpen)
  }

  return (
    <SidebarContext.Provider value={{ isOpen, toggle }}>
      {children}
    </SidebarContext.Provider>
  )
}

export const useSidebar = () => {
  const context = useContext(SidebarContext)
  if (context === undefined) {
    throw new Error("useSidebar must be used within a SidebarProvider")
  }
  return context
} 