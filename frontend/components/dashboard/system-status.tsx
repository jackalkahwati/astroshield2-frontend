import { DataCard } from "@/components/ui/data-card"
import { StatusBadge } from "@/components/ui/status-badge"
import { ProgressIndicator } from "@/components/ui/progress-indicator"

const statusItems = [
  { name: "Satellite Network", status: "nominal", value: 100 },
  { name: "Ground Stations", status: "nominal", value: 100 },
  { name: "Data Processing", status: "warning", value: 87 },
  { name: "Communication Links", status: "nominal", value: 100 },
]

export function SystemStatus() {
  return (
    <DataCard title="System Status" className="w-full">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statusItems.map((item) => (
          <div key={item.name} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{item.name}</span>
              <StatusBadge status={item.status as "nominal" | "warning" | "error"}>
                {item.status.toUpperCase()}
              </StatusBadge>
            </div>
            <ProgressIndicator
              value={item.value}
              status={item.status as "nominal" | "warning" | "error"}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </DataCard>
  )
}

