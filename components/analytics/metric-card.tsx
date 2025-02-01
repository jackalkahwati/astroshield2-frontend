import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface MetricCardProps {
  title: string
  value: string
  subtitle: string
  valueColor?: string
  className?: string
}

export function MetricCard({ title, value, subtitle, valueColor = "text-foreground", className }: MetricCardProps) {
  return (
    <Card className={className}>
      <CardContent className="p-6">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className={cn("text-2xl font-bold", valueColor)}>{value}</p>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
      </CardContent>
    </Card>
  )
}

