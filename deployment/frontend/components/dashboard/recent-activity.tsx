"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { getRelativeTime } from "@/lib/utils/date"

const events = [
  {
    id: 1,
    type: "alert",
    message: "Collision risk detected for AstroShield-1",
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
  },
  {
    id: 2,
    type: "info",
    message: "Scheduled maintenance completed for AstroShield-3",
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
  },
  {
    id: 3,
    type: "success",
    message: "Successful maneuver executed for AstroShield-2",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 4,
    type: "success",
    message: "Threat level normalized for AstroShield-2",
    timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 5,
    type: "info",
    message: "New telemetry data received from all satellites",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
]

export function RecentActivity() {
  return (
    <div className="space-y-8">
      {events.map((event) => (
        <div key={event.id} className="flex items-center">
          <Avatar className="h-9 w-9">
            <AvatarImage src={`https://avatar.vercel.sh/event-${event.id}.png`} alt="Avatar" />
            <AvatarFallback>AS</AvatarFallback>
          </Avatar>
          <div className="ml-4 space-y-1">
            <p className="text-sm font-medium leading-none">{event.message}</p>
            <p className="text-sm text-muted-foreground">
              {getRelativeTime(event.timestamp)}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
} 