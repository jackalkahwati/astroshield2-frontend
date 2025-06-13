'use client'

import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
  ReferenceArea,
  Scatter,
  ScatterChart,
  Cell,
  BarChart,
  Bar
} from 'recharts'
import { AlertTriangle, CheckCircle, Clock, Target, Zap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { HEX_COLORS, getStatusColor, STANDARD_CHART_CONFIG } from '@/lib/chart-colors'

// Timeline/Event Chart (Gantt-like) - Preferred by space operators
interface TimelineEvent {
  name: string
  start: number
  duration: number
  type: 'maneuver' | 'window' | 'conjunction' | 'maintenance' | 'threat'
  status: 'scheduled' | 'active' | 'completed' | 'warning' | 'critical'
  details?: string
}

interface TimelineChartProps {
  events: TimelineEvent[]
  timeRange: [number, number]
  height?: number
}

export const TimelineChart: React.FC<TimelineChartProps> = ({ 
  events, 
  timeRange, 
  height = 300 
}) => {
  const getEventColor = (type: string, status: string) => {
    // Use bright colors ONLY for critical statuses
    if (status === 'critical') return HEX_COLORS.alerts.critical
    if (status === 'warning') return HEX_COLORS.alerts.warning
    
    // Use muted colors for normal operations
    switch (type) {
      case 'maneuver': return HEX_COLORS.status.info      // Muted blue
      case 'window': return HEX_COLORS.status.good        // Muted green
      case 'conjunction': return HEX_COLORS.status.caution // Muted amber
      case 'maintenance': return HEX_COLORS.status.neutral // Gray
      case 'threat': return HEX_COLORS.alerts.urgent      // Dark red (only for threats)
      default: return HEX_COLORS.primary                  // Default muted color
    }
  }

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'maneuver': return <Target className="h-3 w-3" />
      case 'window': return <Clock className="h-3 w-3" />
      case 'conjunction': return <AlertTriangle className="h-3 w-3" />
      case 'threat': return <Zap className="h-3 w-3" />
      default: return <CheckCircle className="h-3 w-3" />
    }
  }

  return (
    <div className="space-y-4">
      <div className="h-16 relative bg-chart-primary border border-chart rounded" 
           style={{ backgroundColor: HEX_COLORS.background.primary, borderColor: HEX_COLORS.border }}>
        {/* Time axis */}
        <div className="absolute bottom-0 left-0 right-0 h-6 border-t"
             style={{ backgroundColor: HEX_COLORS.background.secondary, borderColor: HEX_COLORS.grid }}>
          <div className="flex justify-between items-center px-4 h-full text-xs" 
               style={{ color: HEX_COLORS.axis }}>
            <span>T-{timeRange[0]}h</span>
            <span>Now</span>
            <span>T+{timeRange[1]}h</span>
          </div>
        </div>
        
        {/* Timeline grid */}
        <div className="absolute inset-0 bottom-6">
          {Array.from({ length: 12 }, (_, i) => (
            <div
              key={i}
              className="absolute h-full w-px"
              style={{ 
                left: `${(i / 11) * 100}%`,
                backgroundColor: HEX_COLORS.grid
              }}
            />
          ))}
        </div>
      </div>

      {/* Event lanes */}
      <div className="space-y-2">
        {events.map((event, index) => {
          const startPercent = ((event.start - timeRange[0]) / (timeRange[1] - timeRange[0])) * 100
          const widthPercent = (event.duration / (timeRange[1] - timeRange[0])) * 100
          
          // Better positioning logic to prevent overlaps
          const safeStartPercent = Math.max(startPercent, 35) // More left margin
          const maxWidth = 55 // Reserve more space for badge on right
          const safeWidthPercent = Math.min(widthPercent, maxWidth - safeStartPercent)
          
          return (
            <div key={index} className="relative h-16 border border-gray-800 rounded flex items-center"
                 style={{ backgroundColor: HEX_COLORS.background.secondary, borderColor: HEX_COLORS.border }}>
              
              {/* Event name and icon - fixed width with proper spacing */}
              <div className="absolute left-3 top-2 flex items-center gap-2 text-white w-44 z-10">
                {getEventIcon(event.type)}
                <span className="text-sm font-medium truncate">{event.name}</span>
              </div>
              
              {/* Event details below name - with spacing */}
              <div className="absolute left-3 bottom-2 text-xs text-gray-400 w-44 truncate z-10">
                {event.details}
              </div>
              
              {/* Event bar - carefully positioned to avoid text overlap */}
              <div
                className="absolute h-6 top-3 rounded flex items-center justify-center text-white text-xs font-medium z-20"
                style={{
                  left: `${safeStartPercent}%`,
                  width: `${Math.max(safeWidthPercent, 10)}%`, // Minimum 10% width for visibility
                  backgroundColor: getEventColor(event.type, event.status),
                  minWidth: '80px', // Increased minimum width
                  maxWidth: '45%'   // Maximum width to prevent badge overlap
                }}
              >
                {safeWidthPercent > 12 && event.status.toUpperCase()} {/* Higher threshold for text */}
              </div>
              
              {/* Status indicator - positioned with more margin to avoid overlap */}
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 z-30">
                <Badge 
                  variant={event.status === 'critical' ? 'destructive' : 
                          event.status === 'warning' ? 'secondary' : 'default'}
                  className="text-xs whitespace-nowrap"
                >
                  {event.type.toUpperCase()}
                </Badge>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Anomaly Detection Chart with Event Flags - Critical for space operators
interface AnomalyDataPoint {
  time: string
  value: number
  anomaly?: boolean
  threshold?: number
  event?: string
  confidence?: number
}

interface AnomalyChartProps {
  data: AnomalyDataPoint[]
  metric: string
  thresholds?: { warning: number; critical: number }
  height?: number
}

export const AnomalyChart: React.FC<AnomalyChartProps> = ({ 
  data, 
  metric, 
  thresholds, 
  height = 300 
}) => {
  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props
    if (payload.anomaly) {
      return (
        <circle 
          cx={cx} 
          cy={cy} 
          r={4} 
          fill={HEX_COLORS.alerts.critical}
          stroke="#FFFFFF" 
          strokeWidth={2}
        />
      )
    }
    return null
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="border rounded p-3"
             style={{ 
               backgroundColor: HEX_COLORS.background.secondary,
               borderColor: HEX_COLORS.border
             }}>
          <p className="text-white font-medium">{label}</p>
          <p style={{ color: HEX_COLORS.status.info }}>
            {metric}: {payload[0].value}
          </p>
          {data.confidence && (
            <p style={{ color: HEX_COLORS.status.caution }}>
              Confidence: {(data.confidence * 100).toFixed(1)}%
            </p>
          )}
          {data.anomaly && (
            <p style={{ color: HEX_COLORS.alerts.critical }} className="font-medium">
              ⚠ ANOMALY DETECTED
            </p>
          )}
          {data.event && (
            <p className="text-gray-300 text-sm">{data.event}</p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={HEX_COLORS.grid} />
        <XAxis 
          dataKey="time" 
          stroke={HEX_COLORS.axis}
          fontSize={12}
          tickFormatter={(value) => new Date(value).toLocaleTimeString()}
        />
        <YAxis stroke={HEX_COLORS.axis} fontSize={12} />
        
        {/* Threshold lines - use bright colors only for critical */}
        {thresholds?.warning && (
          <ReferenceLine 
            y={thresholds.warning} 
            stroke={HEX_COLORS.status.caution}
            strokeDasharray="5 5"
            label={{ value: "WARNING", position: "topRight" }}
          />
        )}
        {thresholds?.critical && (
          <ReferenceLine 
            y={thresholds.critical} 
            stroke={HEX_COLORS.alerts.critical}
            strokeDasharray="5 5"
            label={{ value: "CRITICAL", position: "topRight" }}
          />
        )}
        
        <Tooltip content={<CustomTooltip />} />
        
        {/* Main trend line - muted color */}
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke={HEX_COLORS.status.good}
          strokeWidth={2}
          dot={<CustomDot />}
          activeDot={{ r: 6, fill: HEX_COLORS.status.good }}
        />
        
        {/* Confidence bands */}
        <Area
          type="monotone"
          dataKey="confidence"
          stroke="none"
          fill={HEX_COLORS.status.good}
          fillOpacity={0.1}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// Kill Chain Visualization - Shows detection → decision → engagement timeline
interface KillChainStep {
  stage: string
  startTime: number
  duration: number
  status: 'pending' | 'active' | 'completed' | 'failed'
  details: string
}

interface KillChainProps {
  steps: KillChainStep[]
  totalDuration: number
}

export const KillChainVisualization: React.FC<KillChainProps> = ({ 
  steps, 
  totalDuration 
}) => {
  return (
    <div className="space-y-4">
      {/* Timeline header */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-white">Kill Chain Timeline</h3>
        <Badge variant="outline">
          Total Time: {totalDuration.toFixed(2)}s
        </Badge>
      </div>

      {/* Swimlane visualization */}
      <div className="space-y-4">
        {steps.map((step, index) => {
          const startPercent = (step.startTime / totalDuration) * 100
          const widthPercent = (step.duration / totalDuration) * 100
          const statusColor = getStatusColor(step.status)
          
          return (
            <div key={index} className="relative">
              {/* Stage label */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <div className="w-4 h-4 rounded-full mr-3" 
                       style={{ backgroundColor: statusColor }} />
                  <span className="text-white font-medium">{step.stage}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-gray-400 text-sm">
                    {step.duration.toFixed(2)}s
                  </span>
                  <Badge 
                    variant={step.status === 'completed' ? 'default' : 
                            step.status === 'active' ? 'secondary' : 'outline'}
                    className="text-xs"
                  >
                    {step.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
              
              {/* Timeline bar with proper spacing */}
              <div className="relative h-12 border border-gray-700 rounded mb-3 mt-6"
                   style={{ backgroundColor: HEX_COLORS.background.primary, borderColor: HEX_COLORS.border }}>
                {/* Time markers - positioned above bar with more spacing */}
                <div className="absolute -top-5 left-0 text-xs text-gray-500">
                  {step.startTime.toFixed(1)}s
                </div>
                <div className="absolute -top-5 right-0 text-xs text-gray-500">
                  {(step.startTime + step.duration).toFixed(1)}s
                </div>
                
                <div
                  className="absolute h-full rounded flex items-center justify-center"
                  style={{
                    left: `${startPercent}%`,
                    width: `${Math.max(widthPercent, 8)}%`, // Ensure minimum visibility
                    backgroundColor: statusColor
                  }}
                >
                  {/* Only show status text if bar is wide enough */}
                  {widthPercent > 18 && (
                    <span className="text-white text-xs font-medium px-2 truncate">
                      {step.status.toUpperCase()}
                    </span>
                  )}
                </div>
              </div>
              
              {/* Details */}
              <div className="text-sm text-gray-400 pl-7">
                {step.details}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Subsystem Telemetry Chart - Multi-line plots for detailed monitoring
interface TelemetryDataPoint {
  timestamp: string
  [key: string]: string | number
}

interface SubsystemTelemetryProps {
  data: TelemetryDataPoint[]
  metrics: Array<{
    key: string
    name: string
    color?: string  // Now optional, will use standardized colors if not provided
    unit?: string
  }>
  height?: number
}

export const SubsystemTelemetryChart: React.FC<SubsystemTelemetryProps> = ({ 
  data, 
  metrics, 
  height = 300 
}) => {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={HEX_COLORS.grid} />
        <XAxis 
          dataKey="timestamp" 
          stroke={HEX_COLORS.axis}
          fontSize={12}
          tickFormatter={(value) => new Date(value).toLocaleTimeString()}
        />
        <YAxis stroke={HEX_COLORS.axis} fontSize={12} />
        <Tooltip 
          contentStyle={STANDARD_CHART_CONFIG.tooltip.contentStyle}
        />
        
        {metrics.map((metric, index) => (
          <Line
            key={metric.key}
            type="monotone"
            dataKey={metric.key}
            stroke={metric.color || HEX_COLORS.lines[index % HEX_COLORS.lines.length]}
            strokeWidth={2}
            dot={false}
            name={`${metric.name} ${metric.unit || ''}`}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

// Mission Planning Gantt Chart - For scheduling operations
interface MissionTask {
  id: string
  name: string
  start: Date
  end: Date
  type: 'maneuver' | 'observation' | 'communication' | 'maintenance'
  priority: 'low' | 'medium' | 'high' | 'critical'
  dependencies?: string[]
}

interface MissionGanttProps {
  tasks: MissionTask[]
  timeWindow: [Date, Date]
}

export const MissionGanttChart: React.FC<MissionGanttProps> = ({ 
  tasks, 
  timeWindow 
}) => {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return HEX_COLORS.alerts.critical
      case 'high': return HEX_COLORS.alerts.warning
      case 'medium': return HEX_COLORS.status.caution
      default: return HEX_COLORS.status.neutral
    }
  }

  const totalDuration = timeWindow[1].getTime() - timeWindow[0].getTime()

  return (
    <div className="space-y-3">
      {tasks.map((task, index) => {
        const startPercent = ((task.start.getTime() - timeWindow[0].getTime()) / totalDuration) * 100
        const widthPercent = ((task.end.getTime() - task.start.getTime()) / totalDuration) * 100
        
        // Apply same safe positioning logic
        const safeStartPercent = Math.max(startPercent, 35)
        const maxWidth = 50
        const safeWidthPercent = Math.min(widthPercent, maxWidth - safeStartPercent)
        
        return (
          <div key={task.id} className="relative h-14 border border-gray-800 rounded flex items-center"
               style={{ backgroundColor: HEX_COLORS.background.secondary, borderColor: HEX_COLORS.border }}>
            <div className="absolute left-3 text-white text-sm font-medium w-40 truncate z-10">
              {task.name}
            </div>
            
            <div
              className="absolute h-6 top-4 rounded flex items-center justify-center text-white text-xs z-20"
              style={{
                left: `${safeStartPercent}%`,
                width: `${Math.max(safeWidthPercent, 8)}%`,
                backgroundColor: getPriorityColor(task.priority),
                minWidth: '60px',
                maxWidth: '40%'
              }}
            >
              {safeWidthPercent > 12 && task.type.toUpperCase()}
            </div>
            
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 z-30">
              <Badge variant={task.priority === 'critical' ? 'destructive' : 'outline'} className="text-xs whitespace-nowrap">
                {task.priority.toUpperCase()}
              </Badge>
            </div>
          </div>
        )
      })}
    </div>
  )
} 