import { ArrowDown, ArrowRight, ArrowUp } from "lucide-react"

interface ProgressMetricProps {
  title: string
  value: number
  trend: "improving" | "stable" | "declining"
}

export function ProgressMetric({ title, value, trend }: ProgressMetricProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">{title}</span>
        <span className="text-2xl font-bold">{value}%</span>
      </div>
      <div className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Trend:</span>
        <span className="flex items-center gap-1">
          {trend === "improving" && (
            <>
              <ArrowUp className="h-4 w-4 text-success" />
              <span className="text-success">improving</span>
            </>
          )}
          {trend === "stable" && (
            <>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <span>stable</span>
            </>
          )}
          {trend === "declining" && (
            <>
              <ArrowDown className="h-4 w-4 text-destructive" />
              <span className="text-destructive">declining</span>
            </>
          )}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div className="h-full bg-primary transition-all duration-500" style={{ width: `${value}%` }} />
      </div>
    </div>
  )
}

