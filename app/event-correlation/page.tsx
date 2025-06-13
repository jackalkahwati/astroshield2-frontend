"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import OrbitalTimelineVisualization from "@/components/analytics/orbital-timeline-visualization"

export default function EventCorrelationPage() {
  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Correlated Events", value: "15" },
    { title: "Active Workflows", value: "4" },
    { title: "Bottlenecks", value: "2" },
    { title: "Processing Rate", value: "89%" },
  ]

  // Event correlation data for table
  const eventData = [
    {
      eventId: "EVT-001",
      type: "Launch Detection",
      subsystem: "SS0",
      status: "correlated",
      parentEvents: 3,
      childEvents: 2,
      timestamp: "2025-01-23T10:15:00Z"
    },
    {
      eventId: "EVT-002",
      type: "State Vector Update",
      subsystem: "SS2",
      status: "processing",
      parentEvents: 1,
      childEvents: 0,
      timestamp: "2025-01-23T10:12:00Z"
    },
    {
      eventId: "EVT-003",
      type: "Proximity Alert",
      subsystem: "SS5",
      status: "correlated",
      parentEvents: 2,
      childEvents: 4,
      timestamp: "2025-01-23T10:08:00Z"
    },
    {
      eventId: "EVT-004",
      type: "Maneuver Execution",
      subsystem: "SS3",
      status: "bottleneck",
      parentEvents: 1,
      childEvents: 0,
      timestamp: "2025-01-23T10:05:00Z"
    },
    {
      eventId: "EVT-005",
      type: "Threat Assessment",
      subsystem: "SS6",
      status: "correlated",
      parentEvents: 3,
      childEvents: 1,
      timestamp: "2025-01-23T10:02:00Z"
    },
  ]

  // Workflow summary data
  const workflowData = [
    {
      workflow: "Launch Event Processing",
      status: "active",
      events: 8,
      avgProcessingTime: "2.3s",
      bottlenecks: 0
    },
    {
      workflow: "Proximity Event Chain",
      status: "active",
      events: 12,
      avgProcessingTime: "1.8s",
      bottlenecks: 1
    },
    {
      workflow: "Maneuver Correlation",
      status: "bottleneck",
      events: 3,
      avgProcessingTime: "5.7s",
      bottlenecks: 2
    },
    {
      workflow: "Threat Response Chain",
      status: "active",
      events: 6,
      avgProcessingTime: "1.2s",
      bottlenecks: 0
    },
  ]

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "correlated": case "active": return "default"
      case "processing": return "secondary"
      case "bottleneck": return "destructive"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Event Correlation</h1>
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

      {/* Workflow Status Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Active Workflows</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Workflow</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Events</TableHead>
                <TableHead>Avg Processing Time</TableHead>
                <TableHead>Bottlenecks</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {workflowData.map((workflow) => (
                <TableRow key={workflow.workflow}>
                  <TableCell className="font-medium">{workflow.workflow}</TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(workflow.status)}>
                      {workflow.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>{workflow.events}</TableCell>
                  <TableCell>{workflow.avgProcessingTime}</TableCell>
                  <TableCell>
                    <Badge variant={workflow.bottlenecks > 0 ? "destructive" : "default"}>
                      {workflow.bottlenecks}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Recent Events Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Recent Events</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Event ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Subsystem</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Parent Events</TableHead>
                <TableHead>Child Events</TableHead>
                <TableHead>Timestamp</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {eventData.map((event) => (
                <TableRow key={event.eventId}>
                  <TableCell className="font-medium">{event.eventId}</TableCell>
                  <TableCell>{event.type}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{event.subsystem}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(event.status)}>
                      {event.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>{event.parentEvents}</TableCell>
                  <TableCell>{event.childEvents}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {new Date(event.timestamp).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Orbital Timeline Visualization */}
      <OrbitalTimelineVisualization />
    </div>
  )
} 