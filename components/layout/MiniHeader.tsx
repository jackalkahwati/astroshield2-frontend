"use client"

import React from 'react'
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  Cpu,
  Zap,
  BarChart3,
  RefreshCw
} from "lucide-react"
import { cn } from "@/lib/utils"

interface MetricItem {
  label: string
  value: string | number
  status?: 'success' | 'warning' | 'error' | 'neutral'
  icon?: React.ReactNode
}

interface MiniHeaderProps {
  title: string
  subtitle?: string
  metrics?: MetricItem[]
  actions?: React.ReactNode
  onRefresh?: () => void
  isRefreshing?: boolean
  className?: string
  compact?: boolean
}

export function MiniHeader({
  title,
  subtitle,
  metrics = [],
  actions,
  onRefresh,
  isRefreshing = false,
  className,
  compact = false
}: MiniHeaderProps) {
  
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'success': return 'text-success border-success/20 bg-success/10'
      case 'warning': return 'text-warning border-warning/20 bg-warning/10'
      case 'error': return 'text-error border-error/20 bg-error/10'
      default: return 'text-text-secondary border-border-subtle bg-surface-2'
    }
  }

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'success': return <CheckCircle2 className="h-3 w-3" />
      case 'warning': return <AlertTriangle className="h-3 w-3" />
      case 'error': return <AlertTriangle className="h-3 w-3" />
      default: return <Activity className="h-3 w-3" />
    }
  }

  return (
    <div className={cn(
      "sticky top-0 z-sticky bg-surface-1/95 backdrop-blur-lg border-b border-border-subtle",
      "transition-all duration-200 ease-out",
      compact ? "py-2 px-4" : "py-3 px-6",
      className
    )}>
      <div className="flex items-center justify-between gap-4">
        {/* Title and subtitle */}
        <div className="flex-shrink-0">
          <h1 className={cn(
            "font-semibold text-text-primary",
            compact ? "text-lg" : "text-xl"
          )}>
            {title}
          </h1>
          {subtitle && (
            <p className="text-sm text-text-muted mt-0.5">
              {subtitle}
            </p>
          )}
        </div>

        {/* Metrics */}
        {metrics.length > 0 && (
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="flex items-center gap-2 overflow-x-auto scrollbar-hide">
              {metrics.map((metric, index) => (
                <div
                  key={index}
                  className={cn(
                    "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-xs font-medium whitespace-nowrap",
                    getStatusColor(metric.status)
                  )}
                >
                  {metric.icon || getStatusIcon(metric.status)}
                  <span className="hidden sm:inline">{metric.label}:</span>
                  <span className="font-semibold">{metric.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {onRefresh && (
            <Button
              onClick={onRefresh}
              disabled={isRefreshing}
              variant="ghost"
              size="sm"
              className="btn-ghost h-8 w-8 p-0"
              title="Refresh data"
            >
              <RefreshCw className={cn(
                "h-4 w-4",
                isRefreshing && "animate-spin"
              )} />
            </Button>
          )}
          {actions}
        </div>
      </div>
    </div>
  )
}

// Predefined metric configurations for common use cases
export const MetricPresets = {
  chatMetrics: (messagesCount: number, model: string, confidence?: number): MetricItem[] => [
    {
      label: "Messages",
      value: messagesCount,
      icon: <BarChart3 className="h-3 w-3" />,
      status: 'neutral'
    },
    {
      label: "Model",
      value: model,
      icon: <Cpu className="h-3 w-3" />,
      status: 'success'
    },
    ...(confidence ? [{
      label: "Confidence",
      value: `${(confidence * 100).toFixed(1)}%`,
      icon: <Zap className="h-3 w-3" />,
      status: confidence > 0.8 ? 'success' as const : confidence > 0.6 ? 'warning' as const : 'error' as const
    }] : [])
  ],

  systemMetrics: (status: string, uptime: string, connections: number): MetricItem[] => [
    {
      label: "Status",
      value: status,
      status: status === 'healthy' ? 'success' : status === 'warning' ? 'warning' : 'error'
    },
    {
      label: "Uptime",
      value: uptime,
      status: 'neutral'
    },
    {
      label: "Connections",
      value: connections,
      status: connections > 0 ? 'success' : 'warning'
    }
  ]
}

export default MiniHeader 