"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Laptop, BarChart2, Shield, Activity, Menu, X, Settings, Layout, Satellite, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/ui/theme-toggle"
import { useSidebar } from "@/components/providers/sidebar-provider"
import { cn } from "@/lib/utils"

const items = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: Layout,
  },
  {
    title: "Satellites",
    href: "/tracking",
    icon: Satellite,
  },
  {
    title: "Maneuvers",
    href: "/maneuvers",
    icon: Activity,
  },
  {
    title: "Threats",
    href: "/protection",
    icon: AlertTriangle,
  },
  {
    title: "Analytics",
    href: "/analytics",
    icon: BarChart2,
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function Sidebar() {
  const pathname = usePathname()
  const { expanded, setExpanded } = useSidebar()

  return (
    <>
      {/* Mobile overlay */}
      {expanded && (
        <div 
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={() => setExpanded(false)}
          aria-hidden="true"
        />
      )}
      
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-full flex-col border-r bg-background transition-all duration-300 ease-in-out",
          expanded ? "w-64" : "w-16",
          "md:relative md:z-0"
        )}
      >
        <div className="flex h-16 items-center justify-between border-b px-4">
          <div className={cn("flex items-center", expanded ? "justify-between w-full" : "justify-center")}>
            {expanded ? (
              <div className="flex items-center">
                <Shield className="h-6 w-6 text-primary" />
                <span className="ml-2 font-semibold">AstroShield</span>
              </div>
            ) : (
              <Shield className="h-6 w-6 text-primary" />
            )}
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setExpanded(false)}
              aria-label="Close sidebar"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-auto py-2">
          <nav className="grid gap-1 px-2">
            {items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                aria-current={pathname === item.href ? "page" : undefined}
              >
                <span
                  className={cn(
                    "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
                    pathname === item.href
                      ? "bg-accent text-accent-foreground"
                      : "transparent"
                  )}
                >
                  <item.icon className={cn("mr-2 h-5 w-5", expanded ? "" : "mx-auto")} />
                  {expanded && <span>{item.title}</span>}
                  {!expanded && (
                    <span className="sr-only">{item.title}</span>
                  )}
                </span>
              </Link>
            ))}
          </nav>
        </div>
        <div className="border-t p-4">
          <div className={cn("flex", expanded ? "justify-between" : "justify-center")}>
            {expanded && <ThemeToggle />}
            <Button
              variant="outline"
              size="icon"
              onClick={() => setExpanded(!expanded)}
              aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
              className="h-8 w-8"
            >
              <Menu className="h-4 w-4" />
              <span className="sr-only">
                {expanded ? "Collapse" : "Expand"}
              </span>
            </Button>
          </div>
        </div>
      </div>
      
      {/* Mobile toggle button (only visible when sidebar is collapsed) */}
      {!expanded && (
        <Button
          variant="outline"
          size="icon"
          className="fixed bottom-4 left-4 z-50 md:hidden"
          onClick={() => setExpanded(true)}
          aria-label="Open sidebar"
        >
          <Menu className="h-5 w-5" />
        </Button>
      )}
    </>
  )
}

