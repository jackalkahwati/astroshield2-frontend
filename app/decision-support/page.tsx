"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import OperationalWorkflows from "@/components/decision-support/operational-workflows"

export default function DecisionSupportPage() {
  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Active Recommendations", value: "7" },
    { title: "High Priority", value: "2" },
    { title: "Response Time", value: "1.2s" },
    { title: "Success Rate", value: "94%" },
  ]

  // Active recommendations data
  const recommendationData = [
    {
      id: "REC-001",
      type: "Collision Avoidance",
      priority: "high",
      confidence: 96,
      estimatedDeltaV: "12.5 m/s",
      timeToExecute: "4h 15m",
      status: "pending"
    },
    {
      id: "REC-002",
      type: "Station Keeping",
      priority: "medium",
      confidence: 89,
      estimatedDeltaV: "3.2 m/s",
      timeToExecute: "12h 30m",
      status: "approved"
    },
    {
      id: "REC-003",
      type: "Threat Response",
      priority: "high",
      confidence: 92,
      estimatedDeltaV: "18.7 m/s",
      timeToExecute: "2h 45m",
      status: "executing"
    },
    {
      id: "REC-004",
      type: "Orbit Adjustment",
      priority: "low",
      confidence: 78,
      estimatedDeltaV: "6.1 m/s",
      timeToExecute: "24h 0m",
      status: "pending"
    },
    {
      id: "REC-005",
      type: "Emergency Maneuver",
      priority: "critical",
      confidence: 98,
      estimatedDeltaV: "25.3 m/s",
      timeToExecute: "1h 20m",
      status: "pending"
    },
  ]

  // Decision workflows
  const workflowData = [
    {
      workflow: "Collision Risk Assessment",
      status: "active",
      decisions: 12,
      avgResponseTime: "1.8s",
      accuracy: "96%"
    },
    {
      workflow: "Maneuver Authorization",
      status: "active",
      decisions: 8,
      avgResponseTime: "2.1s",
      accuracy: "94%"
    },
    {
      workflow: "Threat Mitigation",
      status: "active",
      decisions: 5,
      avgResponseTime: "0.9s",
      accuracy: "98%"
    },
    {
      workflow: "Resource Allocation",
      status: "idle",
      decisions: 3,
      avgResponseTime: "3.2s",
      accuracy: "87%"
    },
  ]

  const getPriorityVariant = (priority: string) => {
    switch (priority) {
      case "critical": return "destructive"
      case "high": return "destructive"
      case "medium": return "secondary"
      case "low": return "default"
      default: return "outline"
    }
  }

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "executing": return "destructive"
      case "approved": return "default"
      case "pending": return "secondary"
      case "active": return "default"
      case "idle": return "outline"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Decision Support</h1>
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

      {/* Decision Workflows */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Decision Workflows</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Workflow</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Decisions</TableHead>
                <TableHead>Avg Response Time</TableHead>
                <TableHead>Accuracy</TableHead>
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
                  <TableCell>{workflow.decisions}</TableCell>
                  <TableCell>{workflow.avgResponseTime}</TableCell>
                  <TableCell>{workflow.accuracy}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Active Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Active Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Priority</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Est. ΔV</TableHead>
                <TableHead>Time to Execute</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recommendationData.map((rec) => (
                <TableRow key={rec.id}>
                  <TableCell className="font-medium">{rec.id}</TableCell>
                  <TableCell>{rec.type}</TableCell>
                  <TableCell>
                    <Badge variant={getPriorityVariant(rec.priority)}>
                      {rec.priority.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>{rec.confidence}%</TableCell>
                  <TableCell>{rec.estimatedDeltaV}</TableCell>
                  <TableCell className="text-sm">{rec.timeToExecute}</TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(rec.status)}>
                      {rec.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Operational Workflows */}
      <OperationalWorkflows />
    </div>
  )
} 