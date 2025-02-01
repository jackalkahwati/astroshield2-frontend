import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { SatelliteTrackingTable } from "./table"

const metrics = [
  { title: "Active Tracks", value: "2" },
  { title: "Alerts", value: "1" },
  { title: "Update Rate", value: "5s" },
]

export function SatelliteTrackingDashboard() {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        {metrics.map((metric) => (
          <Card key={metric.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-white">{metric.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metric.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Active Tracking Data</CardTitle>
        </CardHeader>
        <CardContent>
          <SatelliteTrackingTable />
        </CardContent>
      </Card>
    </div>
  )
}

