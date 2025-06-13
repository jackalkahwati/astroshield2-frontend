"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ViewToggle, useViewToggle } from "@/components/ui/view-toggle"
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
  Cell,
  LineChart,
  Line
} from 'recharts'

export default function ManeuversPage() {
  // Independent toggle states for each table
  const summaryToggle = useViewToggle("graph")
  const recentToggle = useViewToggle("graph")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Total Maneuvers", value: "28" },
    { title: "Completed", value: "23" },
    { title: "Scheduled", value: "5" },
    { title: "Total ΔV", value: "156.3 m/s" },
  ]

  // Maneuver data for table
  const maneuverData = [
    {
      id: "MAN-001",
      satelliteId: "SAT-001",
      type: "Orbital Adjustment",
      status: "completed",
      scheduledTime: "2025-01-22T14:30:00Z",
      deltaV: 12.5,
      duration: 45.2,
      fuelRequired: 2.8
    },
    {
      id: "MAN-002",
      satelliteId: "SAT-002", 
      type: "Station Keeping",
      status: "scheduled",
      scheduledTime: "2025-01-24T16:00:00Z",
      deltaV: 3.2,
      duration: 15.0,
      fuelRequired: 0.8
    },
    {
      id: "MAN-003",
      satelliteId: "SAT-003",
      type: "Collision Avoidance",
      status: "completed",
      scheduledTime: "2025-01-21T08:15:00Z",
      deltaV: 18.7,
      duration: 62.1,
      fuelRequired: 4.2
    },
    {
      id: "MAN-004",
      satelliteId: "SAT-004",
      type: "Orbit Raising",
      status: "scheduled",
      scheduledTime: "2025-01-25T10:45:00Z",
      deltaV: 25.6,
      duration: 90.3,
      fuelRequired: 6.1
    },
    {
      id: "MAN-005",
      satelliteId: "SAT-005",
      type: "De-orbit",
      status: "in-progress",
      scheduledTime: "2025-01-23T12:00:00Z",
      deltaV: 45.8,
      duration: 120.5,
      fuelRequired: 10.2
    },
  ]

  // Summary by type
  const maneuverSummary = [
    {
      type: "Orbital Adjustment",
      count: 8,
      totalDeltaV: 98.4,
      avgFuelRequired: 22.1
    },
    {
      type: "Station Keeping",
      count: 12,
      totalDeltaV: 38.4,
      avgFuelRequired: 9.6
    },
    {
      type: "Collision Avoidance",
      count: 5,
      totalDeltaV: 87.6,
      avgFuelRequired: 19.8
    },
    {
      type: "Orbit Raising",
      count: 2,
      totalDeltaV: 51.2,
      avgFuelRequired: 12.2
    },
    {
      type: "De-orbit",
      count: 1,
      totalDeltaV: 45.8,
      avgFuelRequired: 10.2
    },
  ]

  // Prepare data for charts
  const summaryChartData = maneuverSummary.map(item => ({
    type: item.type.split(' ')[0], // Shortened names for chart
    count: item.count,
    deltaV: item.totalDeltaV,
    fuel: item.avgFuelRequired,
    fill: item.count >= 10 ? '#10B981' : item.count >= 5 ? '#F59E0B' : '#EF4444'
  }))

  const maneuverTimelineData = maneuverData.map((maneuver, index) => ({
    maneuver: maneuver.id,
    deltaV: maneuver.deltaV,
    duration: maneuver.duration,
    fuel: maneuver.fuelRequired,
    status: maneuver.status,
    fill: maneuver.status === 'completed' ? '#10B981' : 
          maneuver.status === 'in-progress' ? '#F59E0B' : '#6B7280'
  }))

  const statusDistribution = [
    { name: 'Completed', value: maneuverData.filter(m => m.status === 'completed').length, fill: '#10B981' },
    { name: 'Scheduled', value: maneuverData.filter(m => m.status === 'scheduled').length, fill: '#6B7280' },
    { name: 'In Progress', value: maneuverData.filter(m => m.status === 'in-progress').length, fill: '#F59E0B' }
  ].filter(item => item.value > 0)

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "completed": return "default"
      case "scheduled": return "secondary"
      case "in-progress": return "outline"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Maneuvers</h1>
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

      {/* Maneuver Summary by Type */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Maneuver Summary by Type</CardTitle>
          <ViewToggle currentView={summaryToggle.viewMode} onViewChange={summaryToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {summaryToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Maneuver Count by Type */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Maneuver Count by Type</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={summaryChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="type" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [value, 'Count']}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {/* Total Delta-V by Type */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Total ΔV by Type (m/s)</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={summaryChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="type" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [`${value} m/s`, 'Total ΔV']}
                    />
                    <Bar dataKey="deltaV" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Maneuver Type</TableHead>
                  <TableHead>Count</TableHead>
                  <TableHead>Total ΔV (m/s)</TableHead>
                  <TableHead>Avg Fuel Required (kg)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {maneuverSummary.map((summary) => (
                  <TableRow key={summary.type}>
                    <TableCell className="font-medium">{summary.type}</TableCell>
                    <TableCell>{summary.count}</TableCell>
                    <TableCell>{summary.totalDeltaV.toFixed(1)}</TableCell>
                    <TableCell>{summary.avgFuelRequired.toFixed(1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Recent Maneuvers */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Recent Maneuvers</CardTitle>
          <ViewToggle currentView={recentToggle.viewMode} onViewChange={recentToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {recentToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Delta-V Timeline */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">ΔV by Maneuver</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={maneuverTimelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="maneuver" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [`${value} m/s`, 'ΔV']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="deltaV" 
                      stroke="#10B981" 
                      strokeWidth={3}
                      dot={{ fill: '#10B981', strokeWidth: 2, r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              {/* Status Distribution Pie Chart */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Status Distribution</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={statusDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                      labelStyle={{ fill: '#F9FAFB', fontSize: 12 }}
                    >
                      {statusDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Maneuver ID</TableHead>
                  <TableHead>Satellite</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>ΔV (m/s)</TableHead>
                  <TableHead>Duration (s)</TableHead>
                  <TableHead>Scheduled Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {maneuverData.map((maneuver) => (
                  <TableRow key={maneuver.id}>
                    <TableCell className="font-medium">{maneuver.id}</TableCell>
                    <TableCell>{maneuver.satelliteId}</TableCell>
                    <TableCell>{maneuver.type}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(maneuver.status)}>
                        {maneuver.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{maneuver.deltaV.toFixed(1)}</TableCell>
                    <TableCell>{maneuver.duration.toFixed(1)}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(maneuver.scheduledTime).toLocaleString()}
                    </TableCell>
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

