"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

export default function TrajectoryPage() {
  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Active Simulations", value: "3" },
    { title: "Objects Tracked", value: "12" },
    { title: "Prediction Accuracy", value: "94%" },
    { title: "Update Rate", value: "5s" },
  ]

  // Trajectory simulation data
  const simulationData = [
    {
      id: "SIM-001",
      object: "SAT-1234",
      model: "NRLMSISE-00",
      status: "running",
      progress: "78%",
      timeRemaining: "2h 15m",
      accuracy: "96.2%"
    },
    {
      id: "SIM-002",
      object: "SAT-2345",
      model: "JB2008",
      status: "completed",
      progress: "100%",
      timeRemaining: "0m",
      accuracy: "94.8%"
    },
    {
      id: "SIM-003",
      object: "DEB-5678",
      model: "DTM-2020",
      status: "queued",
      progress: "0%",
      timeRemaining: "4h 30m",
      accuracy: "N/A"
    },
    {
      id: "SIM-004",
      object: "SAT-3456",
      model: "NRLMSISE-00",
      status: "failed",
      progress: "45%",
      timeRemaining: "N/A",
      accuracy: "N/A"
    },
  ]

  // Current trajectory parameters
  const trajectoryData = [
    {
      object: "SAT-1234",
      altitude: "405.2 km",
      velocity: "7.66 km/s",
      inclination: "51.64°",
      period: "92.8 min",
      nextPass: "2025-01-23T14:30:00Z"
    },
    {
      object: "SAT-2345", 
      altitude: "550.8 km",
      velocity: "7.59 km/s",
      inclination: "97.8°",
      period: "95.4 min",
      nextPass: "2025-01-23T15:45:00Z"
    },
    {
      object: "SAT-3456",
      altitude: "300.1 km",
      velocity: "7.73 km/s",
      inclination: "28.5°",
      period: "90.2 min",
      nextPass: "2025-01-23T13:20:00Z"
    },
    {
      object: "DEB-5678",
      altitude: "750.3 km",
      velocity: "7.45 km/s", 
      inclination: "82.1°",
      period: "99.7 min",
      nextPass: "2025-01-23T16:10:00Z"
    },
  ]

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "running": return "secondary"
      case "completed": return "default"
      case "queued": return "outline"
      case "failed": return "destructive"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Trajectory Analysis</h1>
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

      {/* Active Simulations Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Active Simulations</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Simulation ID</TableHead>
                <TableHead>Object</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Progress</TableHead>
                <TableHead>Time Remaining</TableHead>
                <TableHead>Accuracy</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {simulationData.map((sim) => (
                <TableRow key={sim.id}>
                  <TableCell className="font-medium">{sim.id}</TableCell>
                  <TableCell>{sim.object}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{sim.model}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(sim.status)}>
                      {sim.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>{sim.progress}</TableCell>
                  <TableCell className="text-sm">{sim.timeRemaining}</TableCell>
                  <TableCell>{sim.accuracy}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Current Trajectory Parameters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Current Trajectory Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Object</TableHead>
                <TableHead>Altitude</TableHead>
                <TableHead>Velocity</TableHead>
                <TableHead>Inclination</TableHead>
                <TableHead>Period</TableHead>
                <TableHead>Next Pass</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trajectoryData.map((traj) => (
                <TableRow key={traj.object}>
                  <TableCell className="font-medium">{traj.object}</TableCell>
                  <TableCell>{traj.altitude}</TableCell>
                  <TableCell>{traj.velocity}</TableCell>
                  <TableCell>{traj.inclination}</TableCell>
                  <TableCell>{traj.period}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {new Date(traj.nextPass).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}