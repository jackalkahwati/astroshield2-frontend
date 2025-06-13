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
import RealTimeKafkaStream from "@/components/kafka-monitor/real-time-kafka-stream"

export default function KafkaMonitorPage() {
  // Independent toggle states for each table
  const performanceToggle = useViewToggle("graph")
  const topicsToggle = useViewToggle("graph")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Connected Topics", value: "17/19" },
    { title: "Active Subscriptions", value: "8" },
    { title: "Messages/sec", value: "245" },
    { title: "System Health", value: "98.5%" },
  ]

  // Kafka topics data for table
  const topicData = [
    {
      topic: "sda.ss0.launch-detection",
      subsystem: "SS0",
      status: "healthy",
      messages: 1432,
      lastMessage: "2025-01-23T10:15:00Z",
      lag: 0
    },
    {
      topic: "sda.ss2.state-vector",
      subsystem: "SS2", 
      status: "healthy",
      messages: 5678,
      lastMessage: "2025-01-23T10:14:55Z",
      lag: 2
    },
    {
      topic: "sda.ss5.launch-intent",
      subsystem: "SS5",
      status: "warning",
      messages: 892,
      lastMessage: "2025-01-23T10:13:20Z",
      lag: 15
    },
    {
      topic: "sda.ss5.pez-wez-predictions",
      subsystem: "SS5",
      status: "healthy",
      messages: 234,
      lastMessage: "2025-01-23T10:14:58Z",
      lag: 1
    },
    {
      topic: "astroshield.internal.alerts",
      subsystem: "Internal",
      status: "healthy",
      messages: 98,
      lastMessage: "2025-01-23T10:15:02Z",
      lag: 0
    },
  ]

  // Performance metrics data
  const performanceData = [
    {
      metric: "Throughput",
      value: "1,245 msg/s",
      status: "normal",
      trend: "↑ 5%"
    },
    {
      metric: "Latency",
      value: "45ms avg",
      status: "good", 
      trend: "↓ 2ms"
    },
    {
      metric: "Error Rate",
      value: "0.23%",
      status: "normal",
      trend: "→ 0%"
    },
    {
      metric: "Consumer Lag",
      value: "347 messages",
      status: "warning",
      trend: "↑ 12%"
    },
    {
      metric: "Partition Count",
      value: "76 total",
      status: "normal",
      trend: "→ 0"
    },
  ]

  // Prepare data for charts
  const performanceChartData = [
    { metric: 'Throughput', value: 1245, fill: '#10B981' },
    { metric: 'Latency', value: 45, fill: '#F59E0B' },
    { metric: 'Error Rate', value: 0.23, fill: '#EF4444' },
    { metric: 'Consumer Lag', value: 347, fill: '#F59E0B' },
    { metric: 'Partitions', value: 76, fill: '#6B7280' }
  ]

  const subsystemDistribution = [
    { name: 'SS0', value: topicData.filter(t => t.subsystem === 'SS0').length, fill: '#10B981' },
    { name: 'SS2', value: topicData.filter(t => t.subsystem === 'SS2').length, fill: '#3B82F6' },
    { name: 'SS5', value: topicData.filter(t => t.subsystem === 'SS5').length, fill: '#8B5CF6' },
    { name: 'Internal', value: topicData.filter(t => t.subsystem === 'Internal').length, fill: '#6B7280' }
  ].filter(item => item.value > 0)

  const topicMessagesData = topicData.map(topic => ({
    topic: topic.topic.split('.').pop(), // Get last part of topic name
    messages: topic.messages,
    lag: topic.lag,
    fill: topic.status === 'healthy' ? '#10B981' : '#F59E0B'
  }))

  const healthStatusData = [
    { name: 'Healthy', value: topicData.filter(t => t.status === 'healthy').length, fill: '#10B981' },
    { name: 'Warning', value: topicData.filter(t => t.status === 'warning').length, fill: '#F59E0B' },
    { name: 'Error', value: topicData.filter(t => t.status === 'error').length, fill: '#EF4444' }
  ].filter(item => item.value > 0)

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "healthy": case "normal": case "good": return "default"
      case "warning": return "secondary"
      case "error": case "critical": return "destructive"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Kafka Monitor</h1>
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

      {/* Performance Metrics */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Performance Metrics</CardTitle>
          <ViewToggle currentView={performanceToggle.viewMode} onViewChange={performanceToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {performanceToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Metrics Chart */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">System Performance</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="metric" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis stroke="#9CA3AF" fontSize={12} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Subsystem Distribution */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Topics by Subsystem</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={subsystemDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                      labelStyle={{ fill: '#F9FAFB', fontSize: 12 }}
                    >
                      {subsystemDistribution.map((entry, index) => (
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
                  <TableHead>Metric</TableHead>
                  <TableHead>Value</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Trend</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {performanceData.map((metric) => (
                  <TableRow key={metric.metric}>
                    <TableCell className="font-medium">{metric.metric}</TableCell>
                    <TableCell>{metric.value}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(metric.status)}>
                        {metric.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm">{metric.trend}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Topic Health Status */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Topic Health Status</CardTitle>
          <ViewToggle currentView={topicsToggle.viewMode} onViewChange={topicsToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {topicsToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Message Volume by Topic */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Message Volume by Topic</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topicMessagesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="topic" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis stroke="#9CA3AF" fontSize={12} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                    />
                    <Bar dataKey="messages" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Health Status Distribution */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Health Status Distribution</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={healthStatusData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                      labelStyle={{ fill: '#F9FAFB', fontSize: 12 }}
                    >
                      {healthStatusData.map((entry, index) => (
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
                  <TableHead>Topic</TableHead>
                  <TableHead>Subsystem</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Messages</TableHead>
                  <TableHead>Consumer Lag</TableHead>
                  <TableHead>Last Message</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topicData.map((topic) => (
                  <TableRow key={topic.topic}>
                    <TableCell className="font-medium">{topic.topic}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{topic.subsystem}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(topic.status)}>
                        {topic.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{topic.messages.toLocaleString()}</TableCell>
                    <TableCell>
                      <Badge variant={topic.lag > 10 ? "secondary" : "default"}>
                        {topic.lag}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(topic.lastMessage).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Real-Time Kafka Stream */}
      <RealTimeKafkaStream 
        topics={['ss5.launch.detection', 'ss5.pez-wez.kkv', 'ss5.threat.correlation']}
        maxMessages={50}
        autoScroll={true}
      />
    </div>
  )
} 