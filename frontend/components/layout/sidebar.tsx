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
} from "lucide-react"

const routes = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
  },
  {
    label: "Analytics",
    icon: ChartBar,
    href: "/analytics",
  },
  {
    label: "Maneuvers",
    icon: Rocket,
    href: "/maneuvers",
  },
  {
    label: "Protection",
    icon: Shield,
    href: "/protection",
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
      "fixed inset-y-0 left-0 z-50 flex h-full flex-col border-r bg-background transition-all",
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
          className={cn("h-8 w-8 p-0", !isOpen && "hidden")}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          onClick={toggle}
          variant="ghost"
          className={cn("h-8 w-8 p-0", isOpen && "hidden")}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1 overflow-hidden">
        <div className="space-y-2 p-4">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "flex items-center gap-x-2 rounded-lg px-3 py-2 text-sm font-medium transition-all hover:bg-accent hover:text-accent-foreground",
                pathname === route.href ? "bg-accent text-accent-foreground" : "text-muted-foreground",
                !isOpen && "justify-center"
              )}
            >
              <route.icon className="h-4 w-4" />
              {isOpen && <span>{route.label}</span>}
            </Link>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}

