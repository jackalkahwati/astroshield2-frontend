"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { SatelliteIcon, AlertTriangle, Shield, Activity, Globe, Clock, Target, Radar, RefreshCw } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ViewToggle, useViewToggle } from "@/components/ui/view-toggle"
import { OrbitalMapView } from "@/components/dashboard/orbital-map-view"
import { 
  TimelineChart,
  AnomalyChart,
  KillChainVisualization,
  SubsystemTelemetryChart
} from "@/components/charts/space-operator-charts"
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { HEX_COLORS, STANDARD_CHART_CONFIG, CHART_COLOR_SEQUENCE } from '@/lib/chart-colors'
import { Button } from "@/components/ui/button"

export default function DashboardPage() {
  // Redirect to comprehensive dashboard for rollup view
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.location.href = '/dashboard/comprehensive'
    }
  }, [])

  // View toggle states for different sections
  const timelineToggle = useViewToggle("graph")
  const systemStatusToggle = useViewToggle("list")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { 
      title: "System Health", 
      value: "96%", 
      icon: Shield
    },
    { 
      title: "Active Tracks", 
      value: "47", 
      icon: SatelliteIcon
    },
    { 
      title: "Critical Alerts", 
      value: "2", 
      icon: AlertTriangle
    },
    { 
      title: "Response Time", 
      value: "1.29s", 
      icon: Activity
    }
  ]

  // Timeline data for space operations
  const missionTimeline = [
    {
      name: "ISS Proximity Maneuver",
      start: -2,
      duration: 4,
      type: 'maneuver' as const,
      status: 'active' as const,
      details: "Delta-V: 2.3 m/s"
    },
    {
      name: "STARLINK-4729 Conjunction",
      start: 1,
      duration: 0.5,
      type: 'conjunction' as const,
      status: 'warning' as const,
      details: "Miss distance: 180m"
    },
    {
      name: "Ground Contact Window",
      start: 3,
      duration: 12,
      type: 'window' as const,
      status: 'scheduled' as const,
      details: "Goldstone DSN"
    },
    {
      name: "Potential Threat Assessment",
      start: 0.5,
      duration: 1,
      type: 'threat' as const,
      status: 'critical' as const,
      details: "Unknown object approach"
    }
  ]

  // Mission event distribution for graph view - using muted colors
  const eventDistribution = [
    { name: 'Maneuvers', value: 1, fill: HEX_COLORS.status.info },
    { name: 'Conjunctions', value: 1, fill: HEX_COLORS.status.caution },
    { name: 'Windows', value: 1, fill: HEX_COLORS.status.good },
    { name: 'Threats', value: 1, fill: HEX_COLORS.alerts.critical }
  ]

  // System health data for graph view
  const systemHealthData = [
    { name: 'Satellite Tracking', health: 99.7 },
    { name: 'Event Processing', health: 92 },
    { name: 'Protection Systems', health: 98 },
    { name: 'Maneuver Planning', health: 96.5 },
    { name: 'CCDM Analysis', health: 87.5 }
  ]

  // Kill chain visualization data
  const killChainSteps = [
    {
      stage: "Detection",
      startTime: 0,
      duration: 1.2,
      status: 'completed' as const,
      details: "Radar contact established"
    },
    {
      stage: "Classification", 
      startTime: 1.2,
      duration: 2.8,
      status: 'completed' as const,
      details: "Threat assessment: HIGH"
    },
    {
      stage: "Decision",
      startTime: 4.0,
      duration: 1.5,
      status: 'active' as const,
      details: "Command authorization pending"
    },
    {
      stage: "Engagement",
      startTime: 5.5,
      duration: 3.0,
      status: 'pending' as const,
      details: "Awaiting execute command"
    }
  ]

  // Sample system status data
  const systemStatus = [
    {
      system: "Satellite Tracking",
      health: "99.7%",
      status: "operational",
      metrics: "47 active tracks",
      lastUpdate: "Jun 6, 2025, 6:41 PM"
    },
    {
      system: "Event Processing", 
      health: "92%",
      status: "warning",
      metrics: "8 processing, 3 high threat",
      lastUpdate: "Jun 6, 2025, 6:41 PM"
    },
    {
      system: "Protection Systems",
      health: "98%", 
      status: "operational",
      metrics: "2 active threats",
      lastUpdate: "Jun 6, 2025, 6:41 PM"
    },
    {
      system: "Maneuver Planning",
      health: "96.5%",
      status: "operational", 
      metrics: "5 scheduled, 156.3 m/s total ΔV",
      lastUpdate: "Jun 6, 2025, 6:41 PM"
    },
    {
      system: "CCDM Analysis",
      health: "87.5%",
      status: "warning",
      metrics: "5/8 indicators passing", 
      lastUpdate: "Jun 6, 2025, 6:41 PM"
    }
  ]

  return (
    <div className="space-y-6">
      {/* Simple header following established standard */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Space Operations Center</h1>
      </div>

      {/* Key Metrics - Same pattern as Satellite Tracking */}
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

      {/* 3D Orbital View */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Orbital Situational Awareness</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Fallback View */}
            <div className="lg:col-span-2">
              <div className="h-96 bg-[#0A0E1A] border border-gray-700 rounded relative">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <Globe className="h-16 w-16 mx-auto text-blue-400" />
                    <div className="text-white">
                      <h3 className="text-lg font-semibold mb-2">Satellite Tracking Active</h3>
                      <p className="text-sm text-gray-400 mb-4">Real-time orbital monitoring</p>
                      <div className="grid grid-cols-2 gap-4 text-left max-w-md">
                        <div className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                          <div className="w-3 h-3 rounded-full bg-green-400" />
                          <div>
                            <div className="text-xs font-medium text-white">USA-317</div>
                            <div className="text-xs text-gray-400">408km</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                          <div className="w-3 h-3 rounded-full bg-yellow-400" />
                          <div>
                            <div className="text-xs font-medium text-white">STARLINK-4729</div>
                            <div className="text-xs text-gray-400">547km</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                          <div className="w-3 h-3 rounded-full bg-red-400" />
                          <div>
                            <div className="text-xs font-medium text-white">DEB-001</div>
                            <div className="text-xs text-gray-400">375km</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                          <div className="w-3 h-3 rounded-full bg-red-600" />
                          <div>
                            <div className="text-xs font-medium text-white">COSMOS-1408</div>
                            <div className="text-xs text-gray-400">485km</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Status overlay */}
                <div className="absolute top-4 right-4 bg-[#1A1F2E] border border-gray-700 rounded p-3">
                  <div className="text-xs space-y-1">
                    <div className="flex items-center gap-2">
                      <Radar className="h-3 w-3 text-green-400" />
                      <span className="text-green-400">TRACKING ACTIVE</span>
                    </div>
                    <div className="text-white">Coverage: Global</div>
                    <div className="text-xs text-gray-400">Operational Mode</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Tracking Summary & Object List */}
            <div className="space-y-4">
              
              {/* Quick Stats */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Tracked Objects:</span>
                  <span className="text-white font-medium">47</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Active Tracking:</span>
                  <span className="text-green-400 font-medium">45</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Threats:</span>
                  <span className="text-red-400 font-medium">2</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Primary Asset:</span>
                  <span className="text-blue-400 font-medium text-xs">USA-317 (ISS)</span>
                </div>
              </div>

              <hr className="border-gray-700" />

              {/* Object List */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-white">Critical Objects</h4>
                
                <div className="flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-600" />
                    <div>
                      <div className="text-xs font-medium text-white">COSMOS-1408-DEB</div>
                      <div className="text-xs text-gray-400">485km</div>
                    </div>
                  </div>
                  <Badge variant="destructive" className="text-xs">CRITICAL</Badge>
                </div>

                <div className="flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-400" />
                    <div>
                      <div className="text-xs font-medium text-white">DEB-001</div>
                      <div className="text-xs text-gray-400">375km</div>
                    </div>
                  </div>
                  <Badge variant="destructive" className="text-xs">HIGH</Badge>
                </div>

                <div className="flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-yellow-400" />
                    <div>
                      <div className="text-xs font-medium text-white">STARLINK-4729</div>
                      <div className="text-xs text-gray-400">547km</div>
                    </div>
                  </div>
                  <Badge variant="secondary" className="text-xs">MEDIUM</Badge>
                </div>
              </div>

              {/* Control Actions */}
              <div className="space-y-2">
                <Button variant="outline" size="sm" className="w-full">
                  <RefreshCw className="h-3 w-3 mr-2" />
                  Refresh Tracking
                </Button>
                <Button variant="outline" size="sm" className="w-full">
                  <Globe className="h-3 w-3 mr-2" />
                  Global View
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Mission Timeline */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Mission Timeline & Events</CardTitle>
          <ViewToggle 
            currentView={timelineToggle.viewMode} 
            onViewChange={timelineToggle.setViewMode} 
          />
        </CardHeader>
        <CardContent>
          {timelineToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Timeline Chart */}
              <div>
                <h3 className="text-sm font-medium mb-4 text-white">Event Timeline</h3>
                <TimelineChart 
                  events={missionTimeline} 
                  timeRange={[-6, 18]} 
                  height={250} 
                />
              </div>
              
              {/* Event Distribution */}
              <div>
                <h3 className="text-sm font-medium mb-4 text-white">Event Type Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={eventDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {eventDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={STANDARD_CHART_CONFIG.tooltip.contentStyle}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-white">Event</TableHead>
                  <TableHead className="text-white">Type</TableHead>
                  <TableHead className="text-white">Status</TableHead>
                  <TableHead className="text-white">Time Window</TableHead>
                  <TableHead className="text-white">Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {missionTimeline.map((event, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium text-white">{event.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">
                        {event.type.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={
                        event.status === 'critical' ? 'destructive' : 
                        event.status === 'warning' ? 'secondary' : 'default'
                      }>
                        {event.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-white">
                      T{event.start > 0 ? '+' : ''}{event.start}h to T{event.start + event.duration > 0 ? '+' : ''}{event.start + event.duration}h
                    </TableCell>
                    <TableCell className="text-white">{event.details}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Threat Response Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Threat Response Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <KillChainVisualization 
            steps={killChainSteps}
            totalDuration={8.5}
          />
        </CardContent>
      </Card>

      {/* System Status */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">System Status Overview</CardTitle>
          <ViewToggle 
            currentView={systemStatusToggle.viewMode} 
            onViewChange={systemStatusToggle.setViewMode} 
          />
        </CardHeader>
        <CardContent>
          {systemStatusToggle.isGraphView ? (
            <div>
              <h3 className="text-sm font-medium mb-4 text-white">System Health Performance (%)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={systemHealthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={HEX_COLORS.grid} />
                  <XAxis 
                    dataKey="name" 
                    stroke={HEX_COLORS.axis}
                    fontSize={12}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis stroke={HEX_COLORS.axis} fontSize={12} />
                  <Tooltip 
                    contentStyle={STANDARD_CHART_CONFIG.tooltip.contentStyle}
                  />
                  <Bar 
                    dataKey="health" 
                    radius={[4, 4, 0, 0]}
                    fill={HEX_COLORS.status.good}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-white">System</TableHead>
                  <TableHead className="text-white">Health</TableHead>
                  <TableHead className="text-white">Status</TableHead>
                  <TableHead className="text-white">Metrics</TableHead>
                  <TableHead className="text-white">Last Update</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {systemStatus.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium text-white">{item.system}</TableCell>
                    <TableCell className="text-white">{item.health}</TableCell>
                    <TableCell>
                      <Badge 
                        variant={item.status === "operational" ? "default" : "destructive"}
                        className={item.status === "operational" ? "bg-green-500" : "bg-yellow-500"}
                      >
                        {item.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-white">{item.metrics}</TableCell>
                    <TableCell className="text-white">{item.lastUpdate}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 