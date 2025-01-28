import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { getRelativeTime } from "@/lib/utils/date"

export function RecentSatellites() {
  return (
    <div className="space-y-8">
      {recentSatellites.map((satellite) => (
        <div key={satellite.id} className="flex items-center">
          <Avatar className="h-9 w-9">
            <AvatarImage src={satellite.avatar} alt="Satellite" />
            <AvatarFallback>{satellite.name[0]}</AvatarFallback>
          </Avatar>
          <div className="ml-4 space-y-1">
            <p className="text-sm font-medium leading-none">{satellite.name}</p>
            <p className="text-sm text-muted-foreground">{satellite.status}</p>
          </div>
          <div className="ml-auto font-medium">{satellite.lastSeen}</div>
        </div>
      ))}
    </div>
  )
}

const recentSatellites = [
  {
    id: "SAT-001",
    name: "AstroShield-1",
    avatar: "https://avatar.vercel.sh/astroshield-1.png",
    status: "Active",
    lastSeen: getRelativeTime(new Date().toISOString()),
  },
  {
    id: "SAT-002",
    name: "AstroShield-2",
    avatar: "https://avatar.vercel.sh/astroshield-2.png",
    status: "Inactive",
    lastSeen: getRelativeTime(new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()),
  },
  {
    id: "SAT-003",
    name: "AstroShield-3",
    avatar: "https://avatar.vercel.sh/astroshield-3.png",
    status: "Active",
    lastSeen: getRelativeTime(new Date(Date.now() - 5 * 60 * 1000).toISOString()),
  },
  {
    id: "SAT-004",
    name: "AstroShield-4",
    avatar: "https://avatar.vercel.sh/astroshield-4.png",
    status: "Active",
    lastSeen: getRelativeTime(new Date(Date.now() - 60 * 1000).toISOString()),
  },
  {
    id: "SAT-005",
    name: "AstroShield-5",
    avatar: "https://avatar.vercel.sh/astroshield-5.png",
    status: "Maintenance",
    lastSeen: getRelativeTime(new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()),
  },
]

