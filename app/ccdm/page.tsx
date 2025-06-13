"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
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

export default function CCDMPage() {
  // Independent toggle states for each table
  const categorySummaryToggle = useViewToggle("graph")
  const indicatorsToggle = useViewToggle("graph")

  // Simple metrics following the Satellite Tracking pattern
  const metrics = [
    { title: "Total Indicators", value: "8" },
    { title: "Passing", value: "5" },
    { title: "Warnings", value: "2" },
    { title: "Failures", value: "1" },
  ]

  // CCDM indicators data for table
  const ccdmIndicators = [
    {
      name: "Object Stability",
      category: "Stability",
      algorithm: "LSTM Neural Network",
      status: "pass",
      confidence: 92,
      lastUpdated: "2025-01-23T10:15:00Z"
    },
    {
      name: "Stability Changes",
      category: "Stability", 
      algorithm: "Change Point Detection",
      status: "warning",
      confidence: 78,
      lastUpdated: "2025-01-23T10:12:00Z"
    },
    {
      name: "Maneuvers Detected",
      category: "Maneuvers",
      algorithm: "Bi-LSTM with Attention",
      status: "pass",
      confidence: 95,
      lastUpdated: "2025-01-23T10:10:00Z"
    },
    {
      name: "Pattern of Life",
      category: "Maneuvers",
      algorithm: "Temporal Pattern Mining", 
      status: "fail",
      confidence: 94,
      lastUpdated: "2025-01-23T10:08:00Z"
    },
    {
      name: "RF Detection",
      category: "RF Emissions",
      algorithm: "Convolutional Neural Network",
      status: "warning",
      confidence: 89,
      lastUpdated: "2025-01-23T10:05:00Z"
    },
    {
      name: "Subsatellite Deployment",
      category: "RF Emissions",
      algorithm: "Multi-target Tracking",
      status: "pass",
      confidence: 96,
      lastUpdated: "2025-01-23T10:02:00Z"
    },
    {
      name: "ITU/FCC Compliance",
      category: "Compliance",
      algorithm: "Rule-based System",
      status: "pass",
      confidence: 99,
      lastUpdated: "2025-01-23T10:00:00Z"
    },
    {
      name: "Analyst Consensus",
      category: "Compliance",
      algorithm: "Ensemble Voting",
      status: "pass",
      confidence: 87,
      lastUpdated: "2025-01-23T09:58:00Z"
    },
  ]

  // Category summary data
  const categorySummary = [
    {
      category: "Stability Indicators",
      total: 2,
      passing: 1,
      warnings: 1,
      failures: 0,
      avgConfidence: 85
    },
    {
      category: "Maneuver Indicators", 
      total: 2,
      passing: 1,
      warnings: 0,
      failures: 1,
      avgConfidence: 94
    },
    {
      category: "RF Indicators",
      total: 2,
      passing: 1,
      warnings: 1,
      failures: 0,
      avgConfidence: 92
    },
    {
      category: "Compliance Indicators",
      total: 2,
      passing: 2,
      warnings: 0,
      failures: 0,
      avgConfidence: 93
    },
  ]

  // Prepare data for charts
  const categoryChartData = categorySummary.map(item => ({
    category: item.category.replace(' Indicators', ''),
    passing: item.passing,
    warnings: item.warnings,
    failures: item.failures,
    confidence: item.avgConfidence,
    fill: item.avgConfidence >= 90 ? '#10B981' : item.avgConfidence >= 80 ? '#F59E0B' : '#EF4444'
  }))

  const statusDistribution = [
    { name: 'Passing', value: ccdmIndicators.filter(i => i.status === 'pass').length, fill: '#10B981' },
    { name: 'Warnings', value: ccdmIndicators.filter(i => i.status === 'warning').length, fill: '#F59E0B' },
    { name: 'Failures', value: ccdmIndicators.filter(i => i.status === 'fail').length, fill: '#EF4444' }
  ].filter(item => item.value > 0)

  const confidenceData = ccdmIndicators.map(indicator => ({
    name: indicator.name.split(' ')[0], // Shortened for chart
    confidence: indicator.confidence,
    category: indicator.category,
    fill: indicator.status === 'pass' ? '#10B981' : 
          indicator.status === 'warning' ? '#F59E0B' : '#EF4444'
  }))

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "pass": return "default"
      case "warning": return "secondary"
      case "fail": return "destructive"
      default: return "outline"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">CCDM Analysis</h1>
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

      {/* Category Summary */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Category Summary</CardTitle>
          <ViewToggle currentView={categorySummaryToggle.viewMode} onViewChange={categorySummaryToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {categorySummaryToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Category Performance */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Indicators by Category</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="category" 
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
                    <Bar dataKey="passing" stackId="a" fill="#10B981" name="Passing" />
                    <Bar dataKey="warnings" stackId="a" fill="#F59E0B" name="Warnings" />
                    <Bar dataKey="failures" stackId="a" fill="#EF4444" name="Failures" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Average Confidence */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Average Confidence by Category</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="category" 
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
                      formatter={(value) => [`${value}%`, 'Confidence']}
                    />
                    <Bar dataKey="confidence" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Category</TableHead>
                  <TableHead>Total</TableHead>
                  <TableHead>Passing</TableHead>
                  <TableHead>Warnings</TableHead>
                  <TableHead>Failures</TableHead>
                  <TableHead>Avg Confidence</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {categorySummary.map((category) => (
                  <TableRow key={category.category}>
                    <TableCell className="font-medium">{category.category}</TableCell>
                    <TableCell>{category.total}</TableCell>
                    <TableCell>
                      <Badge variant="default">{category.passing}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">{category.warnings}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="destructive">{category.failures}</Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress value={category.avgConfidence} className="flex-1 h-2" />
                        <span className="text-sm">{category.avgConfidence}%</span>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* CCDM Indicators */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">CCDM Indicators</CardTitle>
          <ViewToggle currentView={indicatorsToggle.viewMode} onViewChange={indicatorsToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {indicatorsToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Status Distribution */}
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

              {/* Confidence Levels */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Confidence Levels</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={confidenceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="name" 
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
                      formatter={(value) => [`${value}%`, 'Confidence']}
                    />
                    <Bar dataKey="confidence" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Indicator</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Algorithm</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Last Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ccdmIndicators.map((indicator) => (
                  <TableRow key={indicator.name}>
                    <TableCell className="font-medium">{indicator.name}</TableCell>
                    <TableCell>{indicator.category}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {indicator.algorithm}
                    </TableCell>
                    <TableCell>
                      <Badge variant={getStatusVariant(indicator.status)}>
                        {indicator.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress value={indicator.confidence} className="flex-1 h-2" />
                        <span className="text-sm">{indicator.confidence}%</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(indicator.lastUpdated).toLocaleString()}
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