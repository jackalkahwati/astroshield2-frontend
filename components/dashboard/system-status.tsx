import { DashboardCard } from "@/components/dashboard/dashboard-card"
import { StatusBadge } from "@/components/ui/status-badge"
import { ProgressIndicator } from "@/components/ui/progress-indicator"

const statusItems = [
  { name: "Database", status: "Operational", value: 100 },
  { name: "API", status: "Operational", value: 98 },
  { name: "Telemetry", status: "Degraded", value: 85 },
  { name: "Analytics", status: "Operational", value: 99 },
]

export function SystemStatus() {
  return (
    <DashboardCard title="System Status">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statusItems.map((item) => (
          <div key={item.name} className="space-y-2">
            <div className="text-sm font-medium">{item.name}</div>
            <div className="flex items-center gap-2">
              <div
                className={`h-2 w-2 rounded-full ${
                  item.value >= 90 ? "bg-green-500" : item.value >= 80 ? "bg-yellow-500" : "bg-red-500"
                }`}
              />
              <span className="text-sm text-muted-foreground">{item.status}</span>
            </div>
          </div>
        ))}
      </div>
    </DashboardCard>
  )
}

