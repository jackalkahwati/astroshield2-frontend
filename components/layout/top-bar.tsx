"use client"

import { ModeToggle } from "@/components/mode-toggle"
import { UserNav } from "@/components/user-nav"
import { NotificationsPopover } from "@/components/notifications-popover"
import { SearchDialog } from "@/components/search-dialog"
import { cn } from "@/lib/utils"

interface TopBarProps {
  className?: string
}

export function TopBar({ className }: TopBarProps) {
  return (
    <header
      className={cn(
        "border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
        className,
      )}
    >
      <div className="flex h-16 items-center px-4">
        <div className="ml-auto flex items-center gap-4">
          <SearchDialog />
          <NotificationsPopover />
          <ModeToggle />
          <UserNav />
        </div>
      </div>
    </header>
  )
}

