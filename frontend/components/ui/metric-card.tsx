import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StatusIndicator } from "./status-indicator"
import { cn } from "@/lib/utils"

interface MetricCardProps {
  title: string
  value: string | number
  status?: "nominal" | "warning" | "error"
  icon?: React.ReactNode
  trend?: {
    value: string
    direction: "up" | "down"
  }
  className?: string
}

export function MetricCard({ title, value, status, icon, trend, className }: MetricCardProps) {
  return (
    <Card className={cn("overflow-hidden transition-all hover:shadow-lg", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          <div className="flex items-center gap-2">
            {status && <StatusIndicator status={status} />}
            {title}
          </div>
        </CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {trend && (
          <p className={cn("mt-2 text-xs", trend.direction === "up" ? "text-success" : "text-destructive")}>
            {trend.value}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

