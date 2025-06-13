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
  LineChart,
  Line,
  ScatterChart,
  Scatter
} from 'recharts'
import PredictiveThreatModel from "@/components/analytics/predictive-threat-model"

export default function AnalyticsPage() {
  // Independent toggle states for each table
  const classificationToggle = useViewToggle("graph")
  const eventsToggle = useViewToggle("graph")

  // Enhanced metrics incorporating expert feedback
  const metrics = [
    { title: "Expert-Validated Classes", value: "47" },
    { title: "Observable Parameters", value: "23" },
    { title: "Uncertainty Quantified", value: "100%" },
    { title: "Expert Confidence", value: "96.8%" },
  ]

  // Enhanced classification system based on expert LinkedIn discussion
  const expertClassificationData = [
    {
      category: "Launch Events",
      subTypes: "ASAT, Non-ASAT, Graveyard Return",
      observables: "Trajectory, Burn Profile, RF Signature",
      detected: 12,
      accuracy: "96.8%",
      uncertainty: "±2.1%",
      expertValidation: "Tom Johnson, Jim Shell"
    },
    {
      category: "Maneuver Events", 
      subTypes: "Drift, Linear Drift, V-bar Hop, R-bar Hop",
      observables: "Delta-V, Duration, Reference Frame",
      detected: 28,
      accuracy: "94.2%",
      uncertainty: "±3.5%",
      expertValidation: "Nathan Parrott, Tom Johnson"
    },
    {
      category: "Proximity Operations",
      subTypes: "RAAN Drift, Conjunction, Docking, Rephase",
      observables: "Miss Distance, Relative Motion, Lighting",
      detected: 15,
      accuracy: "91.5%",
      uncertainty: "±4.2%",
      expertValidation: "Jarrod Brandt, Nathan Parrott"
    },
    {
      category: "Signature Changes",
      subTypes: "RCS, Vmag, RF, Residuals",
      observables: "Brightness, Radar Cross-Section, RF Power",
      detected: 34,
      accuracy: "89.3%",
      uncertainty: "±5.1%",
      expertValidation: "Nathan Parrott, Tom Johnson"
    },
    {
      category: "Discovery/UCT Events",
      subTypes: "Post-Deployment, Parent Separation, New Objects",
      observables: "Track Correlation, Signature Matching",
      detected: 18,
      accuracy: "87.1%",
      uncertainty: "±6.3%",
      expertValidation: "Jim Shell, SDA TAP Lab"
    },
    {
      category: "Mechanism Activity", 
      subTypes: "Deployable, Solar Panel, Antenna Extension",
      observables: "RCS Change, Attitude Anomaly, RF Pattern",
      detected: 9,
      accuracy: "85.7%",
      uncertainty: "±7.2%",
      expertValidation: "Spencer Devins"
    },
    {
      category: "Temporal Maneuver States",
      subTypes: "Detect, Start, Stop, Ongoing",
      observables: "Burn Duration, Thrust Profile, Timing",
      detected: 42,
      accuracy: "92.4%",
      uncertainty: "±3.8%",
      expertValidation: "Tom Johnson"
    }
  ]

  // Expert-validated recent classifications
  const expertValidatedEvents = [
    {
      timestamp: "2025-01-23T14:30:00Z",
      objectId: "GEO-47291",
      classification: "maneuver_linear_drift_ongoing",
      confidence: "94.2%",
      uncertainty: "±4.1%",
      observable: "12hr continuous low-thrust",
      expertNote: "Tom Johnson pattern: GEO stationkeeping"
    },
    {
      timestamp: "2025-01-23T12:15:00Z", 
      objectId: "LEO-99847",
      classification: "proximity_raan_drift",
      confidence: "91.8%",
      uncertainty: "±5.2%",
      observable: "Relative RAAN change 0.1°/day",
      expertNote: "Nathan Parrott classification validated"
    },
    {
      timestamp: "2025-01-23T09:45:00Z",
      objectId: "UCT-12574",
      classification: "discovery_post_deployment", 
      confidence: "87.3%",
      uncertainty: "±8.1%",
      observable: "2 maneuvers post-separation",
      expertNote: "Jim Shell UCT processing criteria"
    },
    {
      timestamp: "2025-01-23T08:20:00Z",
      objectId: "SAT-25544",
      classification: "signature_rcs_change",
      confidence: "89.7%", 
      uncertainty: "±6.4%",
      observable: "RCS increased 15% over 3 orbits",
      expertNote: "Tom Johnson direct observable"
    },
    {
      timestamp: "2025-01-23T06:12:00Z",
      objectId: "GYD-67891",
      classification: "launch_graveyard_return",
      confidence: "96.4%",
      uncertainty: "±2.8%",
      observable: "Return from beyond GeoBelt",
      expertNote: "Thomas Earle resurrection pattern"
    }
  ]

  // Prepare data for charts
  const classificationChartData = expertClassificationData.map(item => ({
    category: item.category.split(' ')[0], // Shortened names
    accuracy: parseFloat(item.accuracy),
    uncertainty: parseFloat(item.uncertainty.replace('±', '').replace('%', '')),
    detected: item.detected,
    fill: parseFloat(item.accuracy) >= 95 ? '#10B981' : 
          parseFloat(item.accuracy) >= 90 ? '#F59E0B' : '#EF4444'
  }))

  const eventsTimelineData = expertValidatedEvents.map((event, index) => ({
    time: index + 1, // Sequential time for display
    confidence: parseFloat(event.confidence),
    uncertainty: parseFloat(event.uncertainty.replace('±', '').replace('%', '')),
    objectId: event.objectId,
    classification: event.classification.split('_')[0], // Shortened
    fill: parseFloat(event.confidence) >= 95 ? '#10B981' : 
          parseFloat(event.confidence) >= 90 ? '#F59E0B' : '#EF4444'
  }))

  const getUncertaintyVariant = (uncertainty: string) => {
    const value = parseFloat(uncertainty.replace('±', '').replace('%', ''))
    if (value <= 3) return "default"
    if (value <= 6) return "secondary"
    return "destructive"
  }

  const getAccuracyVariant = (accuracy: string) => {
    const value = parseFloat(accuracy)
    if (value >= 95) return "default"
    if (value >= 90) return "secondary"
    return "destructive"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Expert-Validated ML Classification</h1>
      </div>

      {/* Enhanced Metrics incorporating expert feedback */}
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

      {/* Expert-Enhanced Classification System */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Expert-Validated Classification System (LinkedIn Discussion)</CardTitle>
          <ViewToggle currentView={classificationToggle.viewMode} onViewChange={classificationToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {classificationToggle.isGraphView ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Accuracy Chart */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Classification Accuracy by Category</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classificationChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="category" 
                      stroke="#9CA3AF"
                      fontSize={10}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      fontSize={12}
                      domain={[80, 100]}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value) => [`${value}%`, 'Accuracy']}
                    />
                    <Bar dataKey="accuracy" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {/* Uncertainty vs Detection Count Scatter */}
              <div className="h-64">
                <h3 className="text-sm font-medium mb-4 text-white">Uncertainty vs Detection Count</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={classificationChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      type="number"
                      dataKey="detected"
                      stroke="#9CA3AF"
                      fontSize={12}
                      name="Detected"
                    />
                    <YAxis 
                      type="number"
                      dataKey="uncertainty"
                      stroke="#9CA3AF"
                      fontSize={12}
                      name="Uncertainty"
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '6px',
                        color: '#F9FAFB'
                      }}
                      formatter={(value, name) => [
                        name === 'uncertainty' ? `±${value}%` : value,
                        name === 'uncertainty' ? 'Uncertainty' : 'Detected'
                      ]}
                    />
                    <Scatter dataKey="uncertainty" fill="#8B5CF6" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Category</TableHead>
                  <TableHead>Expert Sub-Types</TableHead>
                  <TableHead>Observable Parameters</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead>Uncertainty</TableHead>
                  <TableHead>Expert Validation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {expertClassificationData.map((item) => (
                  <TableRow key={item.category}>
                    <TableCell className="font-medium">{item.category}</TableCell>
                    <TableCell className="text-sm">{item.subTypes}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{item.observables}</TableCell>
                    <TableCell>
                      <Badge variant={getAccuracyVariant(item.accuracy)}>
                        {item.accuracy}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getUncertaintyVariant(item.uncertainty)}>
                        {item.uncertainty}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {item.expertValidation}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Expert-Validated Recent Events */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Expert-Validated Recent Classifications</CardTitle>
          <ViewToggle currentView={eventsToggle.viewMode} onViewChange={eventsToggle.setViewMode} />
        </CardHeader>
        <CardContent>
          {eventsToggle.isGraphView ? (
            <div className="h-64">
              <h3 className="text-sm font-medium mb-4 text-white">Recent Event Confidence Timeline</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={eventsTimelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="time"
                    stroke="#9CA3AF"
                    fontSize={12}
                    tickFormatter={(value) => `Event ${value}`}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                    domain={[80, 100]}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '6px',
                      color: '#F9FAFB'
                    }}
                    formatter={(value, name, props) => [
                      name === 'confidence' ? `${value}%` : `±${value}%`,
                      name === 'confidence' ? 'Confidence' : 'Uncertainty'
                    ]}
                    labelFormatter={(value) => `Event ${value}: ${eventsTimelineData[value-1]?.objectId}`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="confidence" 
                    stroke="#10B981" 
                    strokeWidth={3}
                    dot={{ fill: '#10B981', strokeWidth: 2, r: 6 }}
                    name="confidence"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="uncertainty" 
                    stroke="#EF4444" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ fill: '#EF4444', strokeWidth: 2, r: 4 }}
                    name="uncertainty"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Object ID</TableHead>
                  <TableHead>Classification</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Uncertainty</TableHead>
                  <TableHead>Observable Evidence</TableHead>
                  <TableHead>Expert Validation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {expertValidatedEvents.map((event, index) => (
                  <TableRow key={index}>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(event.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-medium">{event.objectId}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono text-xs">
                        {event.classification}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getAccuracyVariant(event.confidence)}>
                        {event.confidence}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getUncertaintyVariant(event.uncertainty)}>
                        {event.uncertainty}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm">{event.observable}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {event.expertNote}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Predictive Threat Modeling */}
      <PredictiveThreatModel />
    </div>
  )
}

