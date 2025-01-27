import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"

interface MetricCardProps {
  title: string
  value?: string | number
  description?: string
  icon?: React.ReactNode
  trend?: {
    value: string
    direction: "up" | "down" | "neutral"
  }
  loading?: boolean
  className?: string
}

export function MetricCard({ 
  title, 
  value, 
  description, 
  icon, 
  trend, 
  loading = false, 
  className
}: MetricCardProps) {
  return (
    <Card className={cn("overflow-hidden transition-all hover:shadow-lg", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        {loading ? (
          <>
            <Skeleton className="h-7 w-20" />
            {description && <Skeleton className="mt-2 h-4 w-24" />}
          </>
        ) : (
          <>
            <div className="text-2xl font-bold">{value}</div>
            {description && <p className="mt-2 text-xs text-muted-foreground">{description}</p>}
            {trend && (
              <p
                className={cn("mt-2 text-xs", {
                  "text-success": trend.direction === "up",
                  "text-destructive": trend.direction === "down",
                  "text-muted-foreground": trend.direction === "neutral",
                })}
              >
                {trend.value}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

