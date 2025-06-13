import { cn } from "@/lib/utils"

interface ProgressBarProps {
  value: number
  max?: number
  status?: "nominal" | "warning" | "error"
  showValue?: boolean
  label?: string
  className?: string
  animate?: boolean
}

export function ProgressBar({
  value,
  max = 100,
  status = "nominal",
  showValue = true,
  label,
  className,
  animate = true,
}: ProgressBarProps) {
  const percentage = (value / max) * 100

  return (
    <div className={cn("space-y-2", className)}>
      {label && (
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">{label}</span>
          {showValue && <span className={cn("font-medium", `text-${status}`)}>{percentage.toFixed(1)}%</span>}
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn("h-full rounded-full transition-all", {
            "bg-success": status === "nominal",
            "bg-warning": status === "warning",
            "bg-destructive": status === "error",
            "transition-[width] duration-500": animate,
          })}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

