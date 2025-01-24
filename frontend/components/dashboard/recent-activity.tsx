import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const recentActivity = [
  {
    id: "1",
    title: "New Satellite Detected",
    description: "AstroShield-1 detected a new satellite in orbit",
    timestamp: "2 hours ago"
  },
  {
    id: "2",
    title: "Maneuver Completed",
    description: "Successful collision avoidance maneuver executed",
    timestamp: "4 hours ago"
  },
  {
    id: "3",
    title: "System Update",
    description: "Software update completed successfully",
    timestamp: "6 hours ago"
  },
  {
    id: "4",
    title: "Alert Resolved",
    description: "Potential collision threat resolved",
    timestamp: "8 hours ago"
  }
]

export function RecentActivity() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Activity</CardTitle>
        <CardDescription>A list of recent events and updates</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          {recentActivity.map((activity) => (
            <div key={activity.id} className="flex items-center">
              <div className="ml-4 space-y-1">
                <p className="text-sm font-medium leading-none">{activity.title}</p>
                <p className="text-sm text-muted-foreground">{activity.description}</p>
                <p className="text-xs text-muted-foreground">{activity.timestamp}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
} 