import { cn } from "@/lib/utils"

interface StatusIndicatorProps {
  status: "nominal" | "warning" | "error"
  pulse?: boolean
  className?: string
}

export function StatusIndicator({ status, pulse = false, className }: StatusIndicatorProps) {
  return (
    <span
      className={cn(
        "inline-block h-2.5 w-2.5 rounded-full",
        {
          "bg-success": status === "nominal",
          "bg-warning": status === "warning",
          "bg-destructive": status === "error",
          "animate-pulse": pulse,
        },
        className,
      )}
    />
  )
}

