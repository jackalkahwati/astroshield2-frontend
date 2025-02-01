"use client"

import { Bell } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

const notifications = [
  { id: 1, message: "New satellite detected", time: "2 minutes ago" },
  { id: 2, message: "Collision risk increased", time: "1 hour ago" },
  { id: 3, message: "System update available", time: "3 hours ago" },
]

export function NotificationsPopover() {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="icon" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-[10px] font-bold text-white flex items-center justify-center">
            {notifications.length}
          </span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="grid gap-4">
          <h4 className="font-medium leading-none">Notifications</h4>
          <div className="grid gap-2">
            {notifications.map((notification) => (
              <div key={notification.id} className="flex items-start gap-4 rounded-md bg-muted p-2">
                <Bell className="mt-1 h-4 w-4" />
                <div className="grid gap-1">
                  <p className="text-sm font-medium leading-none">{notification.message}</p>
                  <p className="text-xs text-muted-foreground">{notification.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

