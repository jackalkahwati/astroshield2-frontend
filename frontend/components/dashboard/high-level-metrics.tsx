import { DataCard } from "@/components/ui/data-card"
import { Satellite, Shield, Clock, Activity } from "lucide-react"

export function HighLevelMetrics() {
  return (
    <div className="space-y-4">
      <DataCard
        title="Total Satellites"
        value="24"
        trend={{ value: "+2 from last month", direction: "up" }}
        icon={<Satellite className="h-4 w-4 text-muted-foreground" />}
      />
      <DataCard
        title="Active Missions"
        value="12"
        trend={{ value: "+1 from last month", direction: "up" }}
        icon={<Activity className="h-4 w-4 text-muted-foreground" />}
      />
      <DataCard
        title="Data Collected"
        value="652 TB"
        trend={{ value: "+78 TB from last month", direction: "up" }}
        icon={<Shield className="h-4 w-4 text-muted-foreground" />}
      />
      <DataCard
        title="Avg. Response Time"
        value="1.2s"
        trend={{ value: "-0.3s from last month", direction: "down" }}
        icon={<Clock className="h-4 w-4 text-muted-foreground" />}
      />
    </div>
  )
}

