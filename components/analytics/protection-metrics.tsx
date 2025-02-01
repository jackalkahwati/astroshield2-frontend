import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ProgressMetric } from "./progress-metric"

export function ProtectionMetrics() {
  return (
    <Card className="bg-card/50">
      <CardHeader>
        <CardTitle>Current Protection Metrics</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <ProgressMetric title="Threat Analysis" value={94.8} trend="stable" />
        <ProgressMetric title="Collision Avoidance" value={95.1} trend="improving" />
        <ProgressMetric title="Debris Tracking" value={94.5} trend="stable" />
        <ProgressMetric title="Protection Status" value={97.9} trend="stable" />
      </CardContent>
    </Card>
  )
}

