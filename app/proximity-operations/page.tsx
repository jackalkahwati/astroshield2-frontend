"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ViewToggle, useViewToggle } from "@/components/ui/view-toggle"
import { 
  TimelineChart,
  AnomalyChart,
  SubsystemTelemetryChart
} from "@/components/charts/space-operator-charts"
import { 
  ScatterChart,
  Scatter,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { HEX_COLORS, STANDARD_CHART_CONFIG, getRiskLevelColor } from '@/lib/chart-colors'

export default function ProximityOperationsPage() {
  // Independent toggle states for each table
  const riskSummaryToggle = useViewToggle("graph")
  const eventsToggle = useViewToggle("graph")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Active Events", value: "4" },
    { title: "Critical Proximities", value: "2" },
    { title: "TCA Predictions", value: "12" },
    { title: "Risk Threshold", value: "1e-5" },
  ]

  // Proximity events data for table
  const proximityData = [
    {
      eventId: "PROX-001",
      primaryObject: "SAT-1234",
      secondaryObject: "DEB-5678",
      tcaTime: "2025-01-24T08:30:00Z",
      missDistance: "125m",
      probability: "1.2e-4",
      riskLevel: "high"
    },
    {
      eventId: "PROX-002",
      primaryObject: "SAT-2345",
      secondaryObject: "SAT-6789",
      tcaTime: "2025-01-24T14:15:00Z",
      missDistance: "890m",
      probability: "3.4e-5",
      riskLevel: "medium"
    },
    {
      eventId: "PROX-003",
      primaryObject: "SAT-3456",
      secondaryObject: "DEB-9012",
      tcaTime: "2025-01-25T02:45:00Z",
      missDistance: "2.1km",
      probability: "8.7e-6",
      riskLevel: "low"
    },
    {
      eventId: "PROX-004",
      primaryObject: "SAT-4567",
      secondaryObject: "SAT-0123",
      tcaTime: "2025-01-25T16:20:00Z",
      missDistance: "45m",
      probability: "2.8e-3",
      riskLevel: "critical"
    },
  ]

  // Risk assessment summary
  const riskSummary = [
    {
      category: "Critical Risk Events",
      count: 1,
      threshold: "< 100m",
      actionRequired: "Immediate maneuver planning"
    },
    {
      category: "High Risk Events",
      count: 1,
      threshold: "< 500m",
      actionRequired: "Enhanced monitoring"
    },
    {
      category: "Medium Risk Events",
      count: 1,
      threshold: "< 1km",
      actionRequired: "Standard tracking"
    },
    {
      category: "Low Risk Events",
      count: 1,
      threshold: "> 1km",
      actionRequired: "Routine observation"
    },
  ]

  // Timeline data for proximity operations planning - Simplified for better layout
  const proximityTimeline = proximityData.map((event, index) => {
    const tcaTime = new Date(event.tcaTime)
    const hoursToTCA = (tcaTime.getTime() - Date.now()) / (1000 * 60 * 60)
    
    return {
      name: event.eventId,
      start: hoursToTCA - 1,
      duration: 2,
      type: 'conjunction' as const,
      status: event.riskLevel === 'critical' ? 'critical' as const : 
              event.riskLevel === 'high' ? 'warning' as const : 'scheduled' as const,
      details: `${event.primaryObject} vs ${event.secondaryObject}`
    }
  })

  // Trajectory monitoring data for anomaly detection
  const trajectoryAnomalyData = Array.from({ length: 30 }, (_, i) => {
    const time = new Date(Date.now() - (29 - i) * 60000).toISOString()
    const baseDistance = 500 + Math.sin(i * 0.2) * 200
    const anomaly = i === 15 || i === 25
    
    return {
      time,
      value: anomaly ? baseDistance - 150 : baseDistance + (Math.random() - 0.5) * 50,
      anomaly,
      confidence: 0.85 + Math.random() * 0.1,
      event: anomaly ? "Trajectory deviation detected" : undefined
    }
  })

  // Proximity risk distribution for pie chart - using standardized colors
  const riskDistribution = [
    { 
      name: 'Critical', 
      value: proximityData.filter(p => p.riskLevel === 'critical').length, 
      fill: HEX_COLORS.alerts.critical 
    },
    { 
      name: 'High', 
      value: proximityData.filter(p => p.riskLevel === 'high').length, 
      fill: HEX_COLORS.alerts.warning 
    },
    { 
      name: 'Medium', 
      value: proximityData.filter(p => p.riskLevel === 'medium').length, 
      fill: HEX_COLORS.status.caution 
    },
    { 
      name: 'Low', 
      value: proximityData.filter(p => p.riskLevel === 'low').length, 
      fill: HEX_COLORS.status.good 
    }
  ].filter(item => item.value > 0)

  // Engagement envelope data for scatter plot
  const engagementData = proximityData.map(event => ({
    eventId: event.eventId,
    missDistance: parseFloat(event.missDistance.replace(/[^\d.]/g, '')),
    probability: parseFloat(event.probability),
    riskLevel: event.riskLevel,
    fill: getRiskLevelColor(event.riskLevel)
  }))

  const getRiskVariant = (risk: string) => {
    switch (risk) {
      case "critical": return "destructive"
      case "high": return "destructive"
      case "medium": return "secondary"
      case "low": return "default"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      {/* Simple header following established standard */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Proximity Operations</h1>
      </div>

      {/* Key Metrics Cards - same pattern as Satellite Tracking */}
      <div className="grid gap-4 md:grid-cols-4">
        {metrics.map((metric) => (
          <Card key={metric.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-white">{metric.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metric.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Proximity Events Timeline - Improved Layout */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Proximity Events Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Timeline visualization */}
            <div className="border border-gray-700 rounded p-4 min-h-[300px]"
                 style={{ backgroundColor: HEX_COLORS.background.primary, borderColor: HEX_COLORS.border }}>
              {/* Time axis header */}
              <div className="flex justify-between items-center mb-4 text-sm border-b pb-2"
                   style={{ color: HEX_COLORS.axis, borderColor: HEX_COLORS.grid }}>
                <span>T-12h</span>
                <span className="font-medium text-white">Now</span>
                <span>T+48h</span>
              </div>
              
              {/* Event lanes with proper spacing */}
              <div className="space-y-4">
                {proximityTimeline.map((event, index) => {
                  const startPercent = ((event.start - (-12)) / (48 - (-12))) * 100
                  const widthPercent = (event.duration / (48 - (-12))) * 100
                  const eventColor = getRiskLevelColor(event.status === 'critical' ? 'critical' : 
                                                     event.status === 'warning' ? 'high' : 'low')
                  
                  return (
                    <div key={index} className="relative">
                      {/* Event info row */}
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: eventColor }}
                          />
                          <span className="text-white font-medium text-sm">{event.name}</span>
                        </div>
                        <Badge variant={
                          event.status === 'critical' ? 'destructive' : 
                          event.status === 'warning' ? 'secondary' : 'default'
                        }>
                          {event.status.toUpperCase()}
                        </Badge>
                      </div>
                      
                      {/* Timeline bar */}
                      <div className="relative h-8 border border-gray-800 rounded"
                           style={{ backgroundColor: HEX_COLORS.background.secondary, borderColor: HEX_COLORS.border }}>
                        <div
                          className="absolute h-full rounded flex items-center justify-center text-white text-xs font-medium"
                          style={{
                            left: `${Math.max(startPercent, 0)}%`,
                            width: `${Math.min(widthPercent, 100 - Math.max(startPercent, 0))}%`,
                            backgroundColor: eventColor,
                            minWidth: '60px'
                          }}
                        >
                          CONJUNCTION
                        </div>
                      </div>
                      
                      {/* Event details */}
                      <div className="mt-1 text-xs text-gray-400">
                        {event.details}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trajectory Monitoring & Risk Assessment */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Trajectory Anomaly Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Trajectory Monitoring</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <AnomalyChart 
                data={trajectoryAnomalyData}
                metric="Miss Distance (m)"
                thresholds={{ warning: 300, critical: 150 }}
                height={300}
              />
            </div>
          </CardContent>
        </Card>

        {/* Risk Assessment Summary */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-white">Risk Assessment Summary</CardTitle>
            <ViewToggle currentView={riskSummaryToggle.viewMode} onViewChange={riskSummaryToggle.setViewMode} />
          </CardHeader>
          <CardContent>
            {riskSummaryToggle.isGraphView ? (
              <div className="h-80">
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-white">Risk Level Distribution</h3>
                </div>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {riskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={STANDARD_CHART_CONFIG.tooltip.contentStyle}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-80 overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-white">Risk Category</TableHead>
                      <TableHead className="text-white">Count</TableHead>
                      <TableHead className="text-white">Distance Threshold</TableHead>
                      <TableHead className="text-white">Action Required</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {riskSummary.map((risk) => (
                      <TableRow key={risk.category}>
                        <TableCell className="font-medium text-white">{risk.category}</TableCell>
                        <TableCell>
                          <Badge variant={risk.count > 0 ? "default" : "outline"}>
                            {risk.count}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-white">{risk.threshold}</TableCell>
                        <TableCell className="text-white text-sm">
                          {risk.actionRequired}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Engagement Envelope Analysis */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Engagement Envelope Analysis</CardTitle>
          <ViewToggle currentView={eventsToggle.viewMode} onViewChange={eventsToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {eventsToggle.isGraphView ? (
            <div className="h-96">
              <div className="mb-4">
                <h3 className="text-sm font-medium text-white">Miss Distance vs Collision Probability</h3>
              </div>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart 
                  data={engagementData}
                  margin={{ top: 20, right: 20, bottom: 80, left: 80 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={HEX_COLORS.grid} />
                  <XAxis 
                    dataKey="missDistance" 
                    stroke={HEX_COLORS.axis}
                    fontSize={12}
                    type="number"
                    name="Miss Distance"
                    label={{ 
                      value: 'Miss Distance (m)', 
                      position: 'insideBottom', 
                      offset: -20,
                      style: { textAnchor: 'middle', fill: HEX_COLORS.axis }
                    }}
                  />
                  <YAxis 
                    dataKey="probability" 
                    stroke={HEX_COLORS.axis}
                    fontSize={12}
                    type="number"
                    name="Probability"
                    tickFormatter={(value) => value.toExponential(1)}
                    label={{ 
                      value: 'Collision Probability', 
                      angle: -90, 
                      position: 'insideLeft',
                      style: { textAnchor: 'middle', fill: HEX_COLORS.axis }
                    }}
                  />
                  <Tooltip 
                    contentStyle={STANDARD_CHART_CONFIG.tooltip.contentStyle}
                    formatter={(value, name) => [
                      name === 'probability' ? value.toExponential(2) : `${value}m`,
                      name === 'probability' ? 'Probability' : 'Miss Distance'
                    ]}
                    labelFormatter={(value, payload) => {
                      if (payload && payload.length > 0) {
                        return `Event: ${payload[0].payload.eventId}`
                      }
                      return value
                    }}
                  />
                  <Scatter 
                    dataKey="probability" 
                    fill="#8884d8"
                    shape={(props) => {
                      const { cx, cy, payload } = props
                      return (
                        <circle 
                          cx={cx} 
                          cy={cy} 
                          r={6} 
                          fill={payload.fill}
                          stroke="white"
                          strokeWidth={2}
                        />
                      )
                    }}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-96 overflow-y-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-white">Event ID</TableHead>
                    <TableHead className="text-white">Primary Object</TableHead>
                    <TableHead className="text-white">Secondary Object</TableHead>
                    <TableHead className="text-white">TCA Time</TableHead>
                    <TableHead className="text-white">Miss Distance</TableHead>
                    <TableHead className="text-white">Probability</TableHead>
                    <TableHead className="text-white">Risk Level</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {proximityData.map((event) => (
                    <TableRow key={event.eventId}>
                      <TableCell className="font-medium text-white">{event.eventId}</TableCell>
                      <TableCell className="text-white">{event.primaryObject}</TableCell>
                      <TableCell className="text-white">{event.secondaryObject}</TableCell>
                      <TableCell className="text-white text-sm">
                        {new Date(event.tcaTime).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-white">{event.missDistance}</TableCell>
                      <TableCell className="text-white font-mono text-sm">{event.probability}</TableCell>
                      <TableCell>
                        <Badge variant={getRiskVariant(event.riskLevel)}>
                          {event.riskLevel.toUpperCase()}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 