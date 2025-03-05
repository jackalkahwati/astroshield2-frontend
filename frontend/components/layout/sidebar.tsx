"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/components/providers/sidebar-provider"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  LayoutDashboard,
  Activity,
  Rocket,
  Shield,
  Settings,
  ChevronLeft,
  ChevronRight,
  Satellite,
  AlertCircle,
  BarChart3,
  Gauge,
  Navigation
} from "lucide-react"

interface SidebarProps extends React.HTMLAttributes<HTMLElement> {
  // Add any additional custom props here
}

const routes = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
  },
  {
    label: "Indicators",
    icon: AlertCircle,
    href: "/indicators",
  },
  {
    label: "Satellite Tracking",
    icon: Satellite,
    href: "/tracking",
  },
  {
    label: "Stability Analysis",
    icon: Gauge,
    href: "/stability",
  },
  {
    label: "Maneuvers",
    icon: Rocket,
    href: "/maneuvers",
  },
  {
    label: "Trajectory Analysis",
    icon: Navigation,
    href: "/trajectory",
  },
  {
    label: "Analytics",
    icon: BarChart3,
    href: "/analytics",
  },
  {
    label: "Settings",
    icon: Settings,
    href: "/settings",
  },
]

export function Sidebar({ className, ...props }: SidebarProps) {
  const pathname = usePathname()
  const { isOpen, toggle } = useSidebar()

  return (
    <aside
      className={cn(
        "h-screen sticky top-0 z-40 border-r border-border bg-background transition-all duration-300",
        isOpen ? "w-64" : "w-16",
        className
      )}
      {...props}
    >
      <div className="flex h-16 items-center justify-between px-4 border-b">
        <div className={cn("flex items-center gap-x-2", !isOpen && "justify-center w-full")}>
          <Shield className="h-8 w-8 text-primary" />
          {isOpen && <span className="text-xl font-bold">AstroShield</span>}
        </div>
        <Button
          onClick={toggle}
          variant="ghost"
          size="icon"
          className={cn("h-8 w-8", !isOpen && "hidden")}
          aria-label="Close sidebar"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          onClick={toggle}
          variant="ghost"
          size="icon"
          className={cn("h-8 w-8", isOpen && "hidden")}
          aria-label="Open sidebar"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="h-[calc(100vh-4rem)]">
        <nav className="space-y-2 p-4">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "flex items-center gap-x-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
                pathname === route.href ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                !isOpen && "justify-center"
              )}
            >
              <route.icon className="h-5 w-5" />
              {isOpen && <span>{route.label}</span>}
            </Link>
          ))}
        </nav>
      </ScrollArea>
    </aside>
  )
}

