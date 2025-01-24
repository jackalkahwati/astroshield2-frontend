"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  BarChart2,
  List,
  Satellite,
  TrendingUp,
  Shield,
  Activity,
  Settings,
  Menu,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useSidebar } from "@/components/ui/sidebar"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

const navigation = [
  {
    name: "Comprehensive",
    href: "/",
    icon: BarChart2,
  },
  {
    name: "Indicators",
    href: "/indicators",
    icon: List,
  },
  {
    name: "Satellite Tracking",
    href: "/tracking",
    icon: Satellite,
  },
  {
    name: "Stability Analysis",
    href: "/stability",
    icon: TrendingUp,
  },
  {
    name: "Maneuvers",
    href: "/maneuvers",
    icon: Shield,
  },
  {
    name: "Analytics",
    href: "/analytics",
    icon: Activity,
  },
]

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname()
  const { isCollapsed, toggleSidebar } = useSidebar()

  return (
    <div
      className={cn(
        "flex h-screen flex-col bg-background transition-all duration-300 ease-in-out",
        isCollapsed ? "w-16" : "w-64",
        className,
      )}
    >
      <div className="flex items-center justify-between px-4 py-4">
        <Link href="/" className={cn("flex items-center gap-2", isCollapsed && "justify-center")}>
          <Shield className="h-6 w-6 text-primary" />
          {!isCollapsed && <span className="text-xl font-bold">AstroShield</span>}
        </Link>
        <Button variant="ghost" size="icon" onClick={toggleSidebar} className="hidden md:flex">
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>
      <div className="flex-1 overflow-auto py-2">
        <nav className="grid gap-1 px-2">
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <TooltipProvider key={item.name}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href={item.href}
                      className={cn(
                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                        isActive
                          ? "bg-secondary text-secondary-foreground"
                          : "text-muted-foreground hover:bg-secondary/50 hover:text-secondary-foreground",
                        isCollapsed && "justify-center",
                      )}
                    >
                      <item.icon className="h-4 w-4" />
                      {!isCollapsed && <span>{item.name}</span>}
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent
                    side="right"
                    className={cn("bg-popover text-popover-foreground", !isCollapsed && "hidden")}
                  >
                    {item.name}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )
          })}
        </nav>
      </div>
      <div className="border-t p-2">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Link
                href="/settings"
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-secondary-foreground",
                  isCollapsed && "justify-center",
                )}
              >
                <Settings className="h-4 w-4" />
                {!isCollapsed && <span>Settings</span>}
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right" className={cn("bg-popover text-popover-foreground", !isCollapsed && "hidden")}>
              Settings
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  )
}

