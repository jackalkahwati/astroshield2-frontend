"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/components/providers/sidebar-provider"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  LayoutDashboard,
  ChartBar,
  Rocket,
  Shield,
  Settings,
  ChevronLeft,
  ChevronRight,
  Satellite,
  Activity,
  BarChart3,
  AlertCircle,
  Gauge
} from "lucide-react"

const routes = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
  },
  {
    label: "Comprehensive",
    icon: Activity,
    href: "/comprehensive",
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

export function Sidebar() {
  const pathname = usePathname()
  const { isOpen, toggle } = useSidebar()

  return (
    <div className={cn(
      "relative h-full flex flex-col border-r bg-background transition-all duration-300",
      isOpen ? "w-64" : "w-[72px]"
    )}>
      <div className="flex h-16 items-center justify-between px-4 py-4">
        <div className={cn("flex items-center gap-x-2", !isOpen && "justify-center w-full")}>
          <Shield className="h-8 w-8 text-primary" />
          {isOpen && <span className="text-xl font-bold">AstroShield</span>}
        </div>
        <Button
          onClick={toggle}
          variant="ghost"
          size="icon"
          className={cn("h-8 w-8", !isOpen && "hidden")}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          onClick={toggle}
          variant="ghost"
          size="icon"
          className={cn("h-8 w-8", isOpen && "hidden")}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-2 p-4">
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
        </div>
      </ScrollArea>
    </div>
  )
}

