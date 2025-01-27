"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"
import { cn } from "@/lib/utils"
import { ReactNode } from "react"
import { Button } from "./button"
import { Slot } from "@radix-ui/react-slot"

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

interface SidebarProps {
  className?: string
  children: ReactNode
}

export function Sidebar({ className, children }: SidebarProps) {
  return (
    <div className={cn("flex min-h-screen flex-col border-r", className)}>
      {children}
    </div>
  )
}

interface SidebarHeaderProps {
  className?: string
  children: ReactNode
}

export function SidebarHeader({ className, children }: SidebarHeaderProps) {
  return (
    <div className={cn("flex h-14 items-center border-b px-4", className)}>
      {children}
    </div>
  )
}

interface SidebarContentProps {
  className?: string
  children: ReactNode
}

export function SidebarContent({ className, children }: SidebarContentProps) {
  return (
    <div className={cn("flex-1 space-y-4 p-4", className)}>
      {children}
    </div>
  )
}

interface SidebarFooterProps {
  className?: string
  children: ReactNode
}

export function SidebarFooter({ className, children }: SidebarFooterProps) {
  return (
    <div className={cn("border-t p-4", className)}>
      {children}
    </div>
  )
}

interface SidebarMenuProps {
  className?: string
  children: ReactNode
}

export function SidebarMenu({ className, children }: SidebarMenuProps) {
  return (
    <nav className={cn("space-y-1", className)}>
      {children}
    </nav>
  )
}

interface SidebarMenuItemProps {
  className?: string
  children: ReactNode
}

export function SidebarMenuItem({ className, children }: SidebarMenuItemProps) {
  return (
    <div className={cn("", className)}>
      {children}
    </div>
  )
}

interface SidebarMenuButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  className?: string
  children: ReactNode
  asChild?: boolean
  isActive?: boolean
}

export function SidebarMenuButton({ 
  className, 
  children, 
  asChild = false,
  isActive = false,
  ...props 
}: SidebarMenuButtonProps) {
  const Comp = asChild ? Slot : "button"
  return (
    <Comp
      className={cn(
        "flex w-full items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        isActive && "bg-primary/10 text-primary",
        className
      )}
      {...props}
    >
      {children}
    </Comp>
  )
}

interface SidebarTriggerProps {
  className?: string
  children?: ReactNode
}

export function SidebarTrigger({ className, children }: SidebarTriggerProps) {
  const { toggleSidebar } = useSidebar()
  return (
    <Button
      variant="ghost"
      size="icon"
      className={cn("", className)}
      onClick={toggleSidebar}
    >
      {children}
    </Button>
  )
}

