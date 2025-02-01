import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface StatusBadgeProps {
  status: "nominal" | "warning" | "error"
  children: React.ReactNode
  tooltip?: string
  className?: string
}

export function StatusBadge({ status, children, tooltip, className }: StatusBadgeProps) {
  const badge = (
    <Badge
      variant="outline"
      className={cn(
        "transition-colors",
        {
          "border-success text-success hover:bg-success/10": status === "nominal",
          "border-warning text-warning hover:bg-warning/10": status === "warning",
          "border-destructive text-destructive hover:bg-destructive/10": status === "error",
        },
        className,
      )}
    >
      {children}
    </Badge>
  )

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{badge}</TooltipTrigger>
          <TooltipContent>
            <p>{tooltip}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return badge
}

