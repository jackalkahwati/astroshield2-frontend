import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface ProgressIndicatorProps {
  value: number
  max?: number
  status?: "nominal" | "warning" | "error"
  showValue?: boolean
  label?: string
  tooltip?: string
  className?: string
  animate?: boolean
}

export function ProgressIndicator({
  value,
  max = 100,
  status = "nominal",
  showValue = true,
  label,
  tooltip,
  className,
  animate = true,
}: ProgressIndicatorProps) {
  const percentage = (value / max) * 100
  const progressBar = (
    <div className={cn("space-y-2", className)}>
      {label && (
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">{label}</span>
          {showValue && (
            <span
              className={cn("font-medium", {
                "text-success": status === "nominal",
                "text-warning": status === "warning",
                "text-destructive": status === "error",
              })}
            >
              {percentage.toFixed(1)}%
            </span>
          )}
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn("h-full transition-all", {
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

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{progressBar}</TooltipTrigger>
          <TooltipContent>
            <p>{tooltip}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return progressBar
}

