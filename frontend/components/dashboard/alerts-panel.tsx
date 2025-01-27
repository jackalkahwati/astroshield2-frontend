import { DashboardCard } from "@/components/dashboard/dashboard-card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, AlertTriangle, Info } from "lucide-react"

const alerts = [
  { type: "error", title: "Critical Error", description: "Satellite SAT-001 communication lost" },
  { type: "warning", title: "Warning", description: "Orbit deviation detected for SAT-003" },
  { type: "info", title: "Information", description: "Routine maintenance scheduled for ground station GS-02" },
]

export function AlertsPanel() {
  return (
    <DashboardCard title="Recent Alerts">
      <div className="space-y-4">
        {alerts.map((alert, index) => (
          <Alert key={index} variant={alert.type as "default" | "destructive"}>
            {alert.type === "error" && <AlertCircle className="h-4 w-4" />}
            {alert.type === "warning" && <AlertTriangle className="h-4 w-4" />}
            {alert.type === "info" && <Info className="h-4 w-4" />}
            <AlertTitle>{alert.title}</AlertTitle>
            <AlertDescription>{alert.description}</AlertDescription>
          </Alert>
        ))}
      </div>
    </DashboardCard>
  )
}

