import type { Metadata } from "next"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StabilityMetricCard } from "@/components/stability/metric-card"

export const metadata: Metadata = {
  title: "Stability Analysis | AstroShield",
  description: "System stability analysis and monitoring",
}

export default function StabilityPage() {
  return (
    <div className="flex-1 space-y-6">
      <Card className="bg-card/50">
        <CardHeader>
          <CardTitle>System Stability Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-lg font-medium">Overall Status:</span>
            <span className="text-lg font-bold text-success">NOMINAL</span>
          </div>
          <div className="text-sm text-muted-foreground">Last Updated: 1/21/2024, 4:00:00 AM</div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <StabilityMetricCard title="Attitude Stability" percentage={95.5} status="NOMINAL" />
        <StabilityMetricCard title="Orbit Stability" percentage={98.2} status="NOMINAL" />
        <StabilityMetricCard title="Thermal Stability" percentage={87.3} status="WARNING" />
        <StabilityMetricCard title="Power Stability" percentage={92.8} status="NOMINAL" />
        <StabilityMetricCard title="Communication Stability" percentage={96.1} status="NOMINAL" />
      </div>
    </div>
  )
}

