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

export default function ProtectionPage() {
  // Independent toggle states for each table
  const systemsToggle = useViewToggle("graph")
  const activitiesToggle = useViewToggle("graph")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Protection Level", value: "98%" },
    { title: "Active Threats", value: "2" },
    { title: "Coverage", value: "92%" },
    { title: "Response Time", value: "1.4s" },
  ]

  // Protection systems data
  const systemsData = [
    {
      system: "Debris Avoidance",
      status: "operational",
      performance: "98%",
      lastCheck: "30 min ago",
      nextMaintenance: "72h",
      coverage: "Global"
    },
    {
      system: "Collision Prevention",
      status: "operational",
      performance: "96%",
      lastCheck: "15 min ago",
      nextMaintenance: "48h",
      coverage: "LEO/MEO"
    },
    {
      system: "Emergency Response",
      status: "operational",
      performance: "99%",
      lastCheck: "5 min ago",
      nextMaintenance: "24h",
      coverage: "All Assets"
    },
    {
      system: "Threat Detection",
      status: "warning",
      performance: "85%",
      lastCheck: "2h ago",
      nextMaintenance: "12h",
      coverage: "GEO Limited"
    },
  ]

  // Recent activities data
  const activitiesData = [
    {
      time: "2025-01-23T12:30:00Z",
      action: "Protection Scan Completed",
      status: "success",
      details: "All systems operational",
      threatLevel: "None"
    },
    {
      time: "2025-01-23T08:15:00Z",
      action: "Debris Avoidance Maneuver",
      status: "warning",
      details: "Emergency maneuver executed",
      threatLevel: "Medium"
    },
    {
      time: "2025-01-23T04:00:00Z",
      action: "System Health Check",
      status: "success",
      details: "All protection systems verified",
      threatLevel: "None"
    },
    {
      time: "2025-01-22T20:45:00Z",
      action: "Threat Mitigation",
      status: "success",
      details: "Potential collision avoided",
      threatLevel: "High"
    },
    {
      time: "2025-01-22T16:30:00Z",
      action: "Coverage Assessment",
      status: "warning",
      details: "GEO coverage reduced",
      threatLevel: "Low"
    },
  ]

  // Prepare data for charts
  const systemsChartData = systemsData.map(system => ({
    system: system.system.split(' ')[0], // Shortened for chart
    performance: parseInt(system.performance.replace('%', '')),
    fill: system.status === 'operational' ? '#10B981' : '#F59E0B'
  }))

  const statusDistribution = [
    { name: 'Operational', value: systemsData.filter(s => s.status === 'operational').length, fill: '#10B981' },
    { name: 'Warning', value: systemsData.filter(s => s.status === 'warning').length, fill: '#F59E0B' },
    { name: 'Maintenance', value: systemsData.filter(s => s.status === 'maintenance').length, fill: '#6B7280' }
  ].filter(item => item.value > 0)

  const threatLevelData = [
    { name: 'None', value: activitiesData.filter(a => a.threatLevel === 'None').length, fill: '#10B981' },
    { name: 'Low', value: activitiesData.filter(a => a.threatLevel === 'Low').length, fill: '#F59E0B' },
    { name: 'Medium', value: activitiesData.filter(a => a.threatLevel === 'Medium').length, fill: '#EF4444' },
    { name: 'High', value: activitiesData.filter(a => a.threatLevel === 'High').length, fill: '#7C2D12' }
  ].filter(item => item.value > 0)

  const activityTimelineData = activitiesData.map((activity, index) => ({
    action: activity.action.split(' ')[0], // Shortened for chart
    time: new Date(activity.time).getHours(),
    threatLevel: activity.threatLevel === 'None' ? 0 : 
                 activity.threatLevel === 'Low' ? 1 :
                 activity.threatLevel === 'Medium' ? 2 : 3,
    fill: activity.status === 'success' ? '#10B981' : '#F59E0B'
  }))

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "operational": case "success": return "default"
      case "warning": return "destructive"
      case "maintenance": return "secondary"
      default: return "outline"
    }
  }

  const getThreatVariant = (threat: string) => {
    switch (threat) {
      case "High": return "destructive"
      case "Medium": return "secondary"
      case "Low": return "outline"
      case "None": return "default"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Protection Status</h1>
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

      {/* Protection Systems Status */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Protection Systems Status</CardTitle>
          <ViewToggle currentView={systemsToggle.viewMode} onViewChange={systemsToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {systemsToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* System Performance */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">System Performance</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={systemsChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="system" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis stroke="#9CA3AF" fontSize={12} domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [`${value}%`, 'Performance']}
                    />
                    <Bar dataKey="performance" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Status Distribution */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">System Status Distribution</h3>
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
                  <TableHead>System</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Performance</TableHead>
                  <TableHead>Last Check</TableHead>
                  <TableHead>Next Maintenance</TableHead>
                  <TableHead>Coverage</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {systemsData.map((system) => (
                  <TableRow key={system.system}>
                    <TableCell className="font-medium">{system.system}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(system.status)}>
                        {system.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{system.performance}</TableCell>
                    <TableCell className="text-sm">{system.lastCheck}</TableCell>
                    <TableCell className="text-sm">{system.nextMaintenance}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{system.coverage}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Recent Protection Activities */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Recent Protection Activities</CardTitle>
          <ViewToggle currentView={activitiesToggle.viewMode} onViewChange={activitiesToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {activitiesToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Activity Timeline */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Activity Timeline (by Hour)</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={activityTimelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#9CA3AF"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#9CA3AF" 
                      fontSize={12}
                      domain={[0, 3]}
                      tickFormatter={(value) => 
                        value === 0 ? 'None' :
                        value === 1 ? 'Low' :
                        value === 2 ? 'Medium' : 'High'
                      }
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [
                        value === 0 ? 'None' :
                        value === 1 ? 'Low' :
                        value === 2 ? 'Medium' : 'High',
                        'Threat Level'
                      ]}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="threatLevel" 
                      stroke="#10B981" 
                      strokeWidth={3}
                      dot={{ fill: '#10B981', strokeWidth: 2, r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Threat Level Distribution */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Threat Level Distribution</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={threatLevelData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                      labelStyle={{ fill: '#F9FAFB', fontSize: 12 }}
                    >
                      {threatLevelData.map((entry, index) => (
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
                  <TableHead>Time</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Details</TableHead>
                  <TableHead>Threat Level</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {activitiesData.map((activity, index) => (
                  <TableRow key={index}>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(activity.time).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-medium">{activity.action}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(activity.status)}>
                        {activity.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm">{activity.details}</TableCell>
                    <TableCell>
                      <Badge variant={getThreatVariant(activity.threatLevel)}>
                        {activity.threatLevel.toUpperCase()}
                      </Badge>
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