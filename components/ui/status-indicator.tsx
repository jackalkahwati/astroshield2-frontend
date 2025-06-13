"use client"

import { cn } from "@/lib/utils"

type StatusType = "success" | "warning" | "error" | "info" | "neutral"

interface StatusIndicatorProps {
  status: StatusType
  className?: string
  pulseEffect?: boolean
  size?: "sm" | "md" | "lg"
  label?: string
  showLabel?: boolean
}

export function StatusIndicator({
  status,
  className,
  pulseEffect = false,
  size = "md",
  label,
  showLabel = false,
}: StatusIndicatorProps) {
  const displayLabel = label || status
  
  const statusColors = {
    success: "bg-green-500",
    warning: "bg-yellow-500",
    error: "bg-red-500",
    info: "bg-blue-500",
    neutral: "bg-gray-500",
  }
  
  const sizeClasses = {
    sm: "h-2 w-2",
    md: "h-3 w-3",
    lg: "h-4 w-4",
  }
  
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span 
        className={cn(
          "rounded-full", 
          statusColors[status], 
          sizeClasses[size],
          pulseEffect && `animate-pulse`,
        )}
        aria-hidden="true"
      />
      {showLabel && (
        <span className="text-sm font-medium capitalize">{displayLabel}</span>
      )}
    </div>
  )
}

export function mapStatusType(status: string): StatusType {
  switch (status.toLowerCase()) {
    case "active":
    case "online":
    case "operational":
    case "success":
      return "success"
    case "warning":
    case "degraded":
    case "partial":
      return "warning"
    case "critical":
    case "error":
    case "failed":
    case "offline":
      return "error"
    case "info":
    case "informational":
      return "info"
    default:
      return "neutral"
  }
}

