import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface StabilityMetricCardProps {
  title: string
  percentage: number
  status: "NOMINAL" | "WARNING" | "ERROR"
}

export function StabilityMetricCard({ title, percentage, status }: StabilityMetricCardProps) {
  return (
    <Card className="bg-card/50">
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="text-4xl font-bold tracking-tight">{percentage.toFixed(1)}%</div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Status:</span>
          <span
            className={cn("text-sm font-bold", {
              "text-success": status === "NOMINAL",
              "text-warning": status === "WARNING",
              "text-destructive": status === "ERROR",
            })}
          >
            {status}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className={cn("h-full transition-all duration-500", {
              "bg-success": status === "NOMINAL",
              "bg-warning": status === "WARNING",
              "bg-destructive": status === "ERROR",
            })}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

