"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

const events = [
  {
    id: "1",
    type: "Threat Detected",
    description: "Potential collision risk identified for SAT-001",
    timestamp: "2 minutes ago",
    status: "warning",
  },
  {
    id: "2",
    type: "Maneuver Completed",
    description: "Successful orbit adjustment for SAT-003",
    timestamp: "10 minutes ago",
    status: "success",
  },
  {
    id: "3",
    type: "System Update",
    description: "Collision avoidance algorithms updated",
    timestamp: "25 minutes ago",
    status: "info",
  },
  {
    id: "4",
    type: "Alert Cleared",
    description: "Threat level normalized for SAT-002",
    timestamp: "1 hour ago",
    status: "success",
  },
  {
    id: "5",
    type: "Data Collection",
    description: "New telemetry data received from all satellites",
    timestamp: "2 hours ago",
    status: "info",
  },
]

export function RecentActivity() {
  return (
    <div className="space-y-8">
      {events.map((event) => (
        <div key={event.id} className="flex items-center">
          <Avatar className="h-9 w-9">
            <AvatarImage src={`/avatars/0${event.id}.png`} alt="Avatar" />
            <AvatarFallback>AS</AvatarFallback>
          </Avatar>
          <div className="ml-4 space-y-1">
            <p className="text-sm font-medium leading-none">{event.type}</p>
            <p className="text-sm text-muted-foreground">
              {event.description}
            </p>
          </div>
          <div className="ml-auto font-medium">
            {event.timestamp}
          </div>
        </div>
      ))}
    </div>
  )
} 