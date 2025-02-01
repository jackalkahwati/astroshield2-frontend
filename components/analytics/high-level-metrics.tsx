import { Card, CardContent } from "@/components/ui/card"
import { MetricCard } from "./metric-card"

interface HighLevelMetricsProps {
  timeRange: string
}

export default function HighLevelMetrics({ timeRange }: HighLevelMetricsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <MetricCard 
        title="Total Conjunctions" 
        value="142" 
        subtitle={`Last ${timeRange}`} 
        className="bg-card/50" 
      />
      <MetricCard
        title="Threats Detected"
        value="22"
        subtitle={`24 Mitigated in ${timeRange}`}
        valueColor="text-warning"
        className="bg-card/50"
      />
      <MetricCard 
        title="Response Time" 
        value="1.29s" 
        subtitle={`Average in ${timeRange}`} 
        className="bg-card/50" 
      />
      <MetricCard
        title="Protection Coverage"
        value="92.0%"
        subtitle={`Overall Coverage in ${timeRange}`}
        valueColor="text-success"
        className="bg-card/50"
      />
    </div>
  )
}

