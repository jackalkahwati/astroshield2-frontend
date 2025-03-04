"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { ModeToggle } from "@/components/mode-toggle"
import { BarChart2, List, Satellite, TrendingUp, Shield, Activity, Settings, Navigation } from "lucide-react"

const menuItems = [
  {
    text: "Comprehensive",
    icon: BarChart2,
    path: "/comprehensive",
    className: "bg-blue-500/10 text-blue-500",
  },
  {
    text: "Indicators",
    icon: List,
    path: "/indicators",
  },
  {
    text: "Satellite Tracking",
    icon: Satellite,
    path: "/tracking",
  },
  {
    text: "Stability Analysis",
    icon: TrendingUp,
    path: "/stability",
  },
  {
    text: "Maneuvers",
    icon: Shield,
    path: "/maneuvers",
  },
  {
    text: "Trajectory Analysis",
    icon: Navigation,
    path: "/trajectory",
  },
  {
    text: "Analytics",
    icon: Activity,
    path: "/analytics",
  },
]

export function AppSidebar() {
  const pathname = usePathname()

  return (
    <Sidebar className="bg-[#1E1E1E] border-r-0">
      <SidebarHeader className="border-b border-white/10">
        <Link href="/" className="flex items-center gap-2 px-4 py-3 text-white">
          <Shield className="h-6 w-6" />
          <span className="font-semibold">AstroShield</span>
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          {menuItems.map((item) => (
            <SidebarMenuItem key={item.path}>
              <SidebarMenuButton
                asChild
                isActive={pathname === item.path}
                className={cn(
                  "h-12 gap-3 text-white/70 hover:text-white hover:bg-white/5",
                  pathname === item.path && "bg-white/5 text-white",
                  item.className,
                )}
              >
                <Link href={item.path}>
                  <item.icon className="h-5 w-5" />
                  {item.text}
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarContent>
      <SidebarFooter className="border-t border-white/10">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="h-12 gap-3 text-white/70 hover:text-white hover:bg-white/5">
              <Link href="/settings">
                <Settings className="h-5 w-5" />
                Settings
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}

