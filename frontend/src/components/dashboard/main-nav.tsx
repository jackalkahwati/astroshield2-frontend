import Link from "next/link"
import { usePathname } from "next/navigation"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

export function MainNav({
  className,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  const pathname = usePathname()

  const routes = [
    {
      href: "/dashboard",
      label: "Overview",
      active: pathname === "/dashboard",
    },
    {
      href: "/dashboard/tracking",
      label: "Satellite Tracking",
      active: pathname === "/dashboard/tracking",
    },
    {
      href: "/dashboard/stability",
      label: "Stability Analysis",
      active: pathname === "/dashboard/stability",
    },
    {
      href: "/dashboard/maneuvers",
      label: "Maneuver Planning",
      active: pathname === "/dashboard/maneuvers",
    },
    {
      href: "/dashboard/physical",
      label: "Physical Properties",
      active: pathname === "/dashboard/physical",
    },
    {
      href: "/dashboard/environmental",
      label: "Environmental",
      active: pathname === "/dashboard/environmental",
    },
    {
      href: "/dashboard/launch",
      label: "Launch Evaluation",
      active: pathname === "/dashboard/launch",
    },
  ]

  return (
    <nav
      className={cn("flex items-center space-x-4 lg:space-x-6", className)}
      {...props}
    >
      {routes.map((route) => (
        <Link
          key={route.href}
          href={route.href}
          className={cn(
            "text-sm font-medium transition-colors hover:text-primary",
            route.active
              ? "text-black dark:text-white"
              : "text-muted-foreground"
          )}
        >
          {route.label}
        </Link>
      ))}
    </nav>
  )
} 