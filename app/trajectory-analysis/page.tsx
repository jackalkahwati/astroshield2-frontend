"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area
} from 'recharts'
import { 
  Satellite, 
  Target, 
  Orbit, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Eye, 
  Navigation, 
  Zap,
  Clock,
  BarChart3,
  Globe,
  MapPin,
  Play,
  Pause,
  RefreshCw
} from "lucide-react"

interface TrajectoryPoint {
  timestamp: string
  position: { x: number; y: number; z: number }
  velocity: { x: number; y: number; z: number }
  altitude: number
  inclination: number
  eccentricity: number
  uncertainty: number
}

interface ManeuverDetection {
  id: string
  timestamp: string
  type: "station_keeping" | "orbit_change" | "avoidance" | "unknown"
  delta_v: number
  confidence: number
  duration: number
  description: string
}

interface TrajectoryAnalysis {
  object_id: string
  analysis_window: string
  total_points: number
  maneuvers_detected: ManeuverDetection[]
  anomalies: any[]
  predictions: any[]
  ccdm_indicators: any[]
}

const mockTrajectoryData: TrajectoryPoint[] = Array.from({ length: 100 }, (_, i) => ({
  timestamp: new Date(Date.now() - (100 - i) * 60000).toISOString(),
  position: {
    x: 6371000 + 400000 * Math.cos(i * 0.1),
    y: 6371000 + 400000 * Math.sin(i * 0.1),
    z: 100000 * Math.sin(i * 0.05)
  },
  velocity: {
    x: 7500 + 100 * Math.sin(i * 0.1),
    y: 200 * Math.cos(i * 0.1),
    z: 50 * Math.sin(i * 0.2)
  },
  altitude: 400 + 50 * Math.sin(i * 0.1),
  inclination: 51.6 + 0.1 * Math.sin(i * 0.05),
  eccentricity: 0.001 + 0.0005 * Math.random(),
  uncertainty: 10 + 5 * Math.random()
}))

const mockManeuvers: ManeuverDetection[] = [
  {
    id: "MAN-001",
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    type: "station_keeping",
    delta_v: 2.3,
    confidence: 0.94,
    duration: 45,
    description: "Station-keeping maneuver detected with nominal delta-V"
  },
  {
    id: "MAN-002",
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    type: "unknown",
    delta_v: 15.7,
    confidence: 0.87,
    duration: 120,
    description: "Unplanned maneuver with significant delta-V - potential CCDM indicator"
  },
  {
    id: "MAN-003",
    timestamp: new Date(Date.now() - 14400000).toISOString(),
    type: "avoidance",
    delta_v: 8.2,
    confidence: 0.91,
    duration: 30,
    description: "Collision avoidance maneuver"
  }
]

export default function TrajectoryAnalysisPage() {
  const [selectedObject, setSelectedObject] = useState("Starlink-2567")
  const [analysisWindow, setAnalysisWindow] = useState("24 Hours")
  const [showUncertainty, setShowUncertainty] = useState(true)
  const [showPredictions, setShowPredictions] = useState(true)
  const [maneuverThreshold, setManeuverThreshold] = useState([5.0])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentTab, setCurrentTab] = useState("overview")

  // Process trajectory data for charts
  const orbitalElementsData = mockTrajectoryData.map((point, index) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    altitude: Math.round(point.altitude),
    inclination: Math.round(point.inclination * 100) / 100,
    eccentricity: Math.round(point.eccentricity * 10000) / 10000,
    uncertainty: showUncertainty ? point.uncertainty : 0
  }))

  const velocityData = mockTrajectoryData.map((point, index) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    velocity: Math.round(Math.sqrt(point.velocity.x ** 2 + point.velocity.y ** 2 + point.velocity.z ** 2)),
    velocity_x: Math.round(point.velocity.x),
    velocity_y: Math.round(point.velocity.y),
    velocity_z: Math.round(point.velocity.z)
  }))

  const analysis: TrajectoryAnalysis = {
    object_id: selectedObject,
    analysis_window: analysisWindow,
    total_points: mockTrajectoryData.length,
    maneuvers_detected: mockManeuvers,
    anomalies: [
      {
        id: "ANOM-001",
        timestamp: new Date(Date.now() - 5400000).toISOString(),
        type: "trajectory_deviation",
        severity: "medium",
        description: "Unexpected trajectory deviation from predicted path"
      }
    ],
    predictions: [
      {
        timestamp: new Date(Date.now() + 3600000).toISOString(),
        confidence: 0.89,
        scenario: "nominal_continuation"
      }
    ],
    ccdm_indicators: [
      {
        type: "maneuver_frequency",
        value: 3,
        threshold: 2,
        status: "alert"
      },
      {
        type: "delta_v_budget",
        value: 26.2,
        expected: 15.0,
        status: "warning"
      }
    ]
  }

  const getManeuverTypeColor = (type: string) => {
    switch (type) {
      case "station_keeping": return "bg-blue-100 text-blue-800"
      case "orbit_change": return "bg-purple-100 text-purple-800"
      case "avoidance": return "bg-green-100 text-green-800"
      case "unknown": return "bg-red-100 text-red-800"
      default: return "bg-gray-100 text-gray-800"
    }
  }

  const runAnalysis = async () => {
    setIsAnalyzing(true)
    // Simulate analysis
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsAnalyzing(false)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold">Trajectory Analysis</h1>
        </div>
        <Button onClick={runAnalysis} disabled={isAnalyzing}>
          {isAnalyzing ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Run Analysis
            </>
          )}
        </Button>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Navigation className="h-5 w-5" />
            Analysis Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="object-select">Target Object</Label>
              <Select value={selectedObject} onValueChange={setSelectedObject}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Starlink-2567">Starlink-2567</SelectItem>
                  <SelectItem value="ISS">ISS</SelectItem>
                  <SelectItem value="Cosmos-2345">Cosmos-2345</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="window-select">Analysis Window</Label>
              <Select value={analysisWindow} onValueChange={setAnalysisWindow}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="6 Hours">6 Hours</SelectItem>
                  <SelectItem value="24 Hours">24 Hours</SelectItem>
                  <SelectItem value="7 Days">7 Days</SelectItem>
                  <SelectItem value="30 Days">30 Days</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label>Maneuver Threshold (m/s)</Label>
              <div className="mt-2">
                <Slider
                  value={maneuverThreshold}
                  onValueChange={setManeuverThreshold}
                  max={20}
                  min={0.1}
                  step={0.1}
                  className="w-full"
                />
                <div className="text-sm text-muted-foreground mt-1">
                  {maneuverThreshold[0]} m/s
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <Switch
                  id="uncertainty"
                  checked={showUncertainty}
                  onCheckedChange={setShowUncertainty}
                />
                <Label htmlFor="uncertainty">Show Uncertainty</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="predictions"
                  checked={showPredictions}
                  onCheckedChange={setShowPredictions}
                />
                <Label htmlFor="predictions">Show Predictions</Label>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Data Points</p>
                <p className="text-2xl font-bold">{analysis.total_points}</p>
              </div>
              <Globe className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Maneuvers Detected</p>
                <p className="text-2xl font-bold">{analysis.maneuvers_detected.length}</p>
              </div>
              <Target className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Anomalies</p>
                <p className="text-2xl font-bold text-yellow-600">{analysis.anomalies.length}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">CCDM Risk</p>
                <p className="text-2xl font-bold text-red-600">
                  {analysis.ccdm_indicators.some(i => i.status === "alert") ? "HIGH" : "MEDIUM"}
                </p>
              </div>
              <Eye className="h-8 w-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={currentTab} onValueChange={setCurrentTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="maneuvers">Maneuvers</TabsTrigger>
          <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="ccdm">CCDM Indicators</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Orbital Elements Trend
                </CardTitle>
                <CardDescription>
                  Real-time tracking of key orbital parameters over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={orbitalElementsData.slice(-20)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#9CA3AF"
                        fontSize={12}
                        tickFormatter={(value) => value.split(':')[0] + ':' + value.split(':')[1]}
                      />
                      <YAxis 
                        yAxisId="altitude"
                        orientation="left"
                        stroke="#3B82F6"
                        fontSize={12}
                      />
                      <YAxis 
                        yAxisId="inclination"
                        orientation="right"
                        stroke="#10B981"
                        fontSize={12}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '6px',
                          color: '#F9FAFB'
                        }}
                        formatter={(value, name) => [
                          `${value} ${name === 'altitude' ? 'km' : name === 'inclination' ? '°' : ''}`,
                          name.charAt(0).toUpperCase() + name.slice(1)
                        ]}
                      />
                      <Line 
                        yAxisId="altitude"
                        type="monotone" 
                        dataKey="altitude" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        dot={false}
                        name="altitude"
                      />
                      <Line 
                        yAxisId="inclination"
                        type="monotone" 
                        dataKey="inclination" 
                        stroke="#10B981" 
                        strokeWidth={2}
                        dot={false}
                        name="inclination"
                      />
                      {showUncertainty && (
                        <Area
                          yAxisId="altitude"
                          type="monotone"
                          dataKey="uncertainty"
                          stroke="#EF4444"
                          fill="#EF4444"
                          fillOpacity={0.1}
                          strokeWidth={1}
                          strokeDasharray="2 2"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Velocity Profile
                </CardTitle>
                <CardDescription>
                  3-axis velocity components and total velocity magnitude
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={velocityData.slice(-20)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#9CA3AF"
                        fontSize={12}
                        tickFormatter={(value) => value.split(':')[0] + ':' + value.split(':')[1]}
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
                        formatter={(value, name) => [
                          `${value} m/s`,
                          name === 'velocity' ? 'Total Velocity' : `Velocity ${name.split('_')[1].toUpperCase()}`
                        ]}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="velocity" 
                        stroke="#8B5CF6" 
                        strokeWidth={3}
                        dot={false}
                        name="velocity"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="velocity_x" 
                        stroke="#EF4444" 
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        dot={false}
                        name="velocity_x"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="velocity_y" 
                        stroke="#10B981" 
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        dot={false}
                        name="velocity_y"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="velocity_z" 
                        stroke="#F59E0B" 
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        dot={false}
                        name="velocity_z"
                      />
                      {mockManeuvers.map((maneuver, index) => {
                        const maneuverTime = new Date(maneuver.timestamp).toLocaleTimeString()
                        return (
                          <ReferenceLine 
                            key={index}
                            x={maneuverTime} 
                            stroke="#DC2626" 
                            strokeDasharray="2 2"
                            label={{ value: `M${index + 1}`, position: "top" }}
                          />
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="maneuvers" className="mt-6">
          <div className="space-y-4">
            {analysis.maneuvers_detected.map((maneuver) => (
              <Card key={maneuver.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{maneuver.id}</CardTitle>
                      <CardDescription>
                        {new Date(maneuver.timestamp).toLocaleString()}
                      </CardDescription>
                    </div>
                    <Badge className={getManeuverTypeColor(maneuver.type)}>
                      {maneuver.type.toUpperCase()}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                      <Label>Delta-V</Label>
                      <p className="text-lg font-semibold">{maneuver.delta_v} m/s</p>
                    </div>
                    <div>
                      <Label>Duration</Label>
                      <p className="text-lg font-semibold">{maneuver.duration} sec</p>
                    </div>
                    <div>
                      <Label>Confidence</Label>
                      <p className="text-lg font-semibold">{Math.round(maneuver.confidence * 100)}%</p>
                    </div>
                    <div>
                      <Label>Status</Label>
                      <Badge variant={maneuver.type === "unknown" ? "destructive" : "secondary"}>
                        {maneuver.type === "unknown" ? "REQUIRES ANALYSIS" : "NOMINAL"}
                      </Badge>
                    </div>
                  </div>
                  <div className="mt-4">
                    <Label>Description</Label>
                    <p className="text-sm text-muted-foreground">{maneuver.description}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="anomalies" className="mt-6">
          <div className="space-y-4">
            {analysis.anomalies.map((anomaly, index) => (
              <Alert key={anomaly.id}>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Trajectory Anomaly Detected</AlertTitle>
                <AlertDescription>
                  <div className="mt-2">
                    <p><strong>Time:</strong> {new Date(anomaly.timestamp).toLocaleString()}</p>
                    <p><strong>Type:</strong> {anomaly.type}</p>
                    <p><strong>Severity:</strong> {anomaly.severity}</p>
                    <p className="mt-2">{anomaly.description}</p>
                  </div>
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="predictions" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Trajectory Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {analysis.predictions.map((prediction, index) => (
                  <div key={index} className="border rounded p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <p className="font-medium">Prediction Horizon</p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(prediction.timestamp).toLocaleString()}
                        </p>
                      </div>
                      <Badge variant="outline">
                        {Math.round(prediction.confidence * 100)}% Confidence
                      </Badge>
                    </div>
                    <p className="text-sm">Scenario: {prediction.scenario}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ccdm" className="mt-6">
          <div className="space-y-4">
            <Alert>
              <Eye className="h-4 w-4" />
              <AlertTitle>CCDM Analysis Summary</AlertTitle>
              <AlertDescription>
                Multiple indicators suggest potential deceptive maneuvering patterns. Manual review recommended.
              </AlertDescription>
            </Alert>
            
            {analysis.ccdm_indicators.map((indicator, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium capitalize">{indicator.type.replace('_', ' ')}</p>
                      <p className="text-2xl font-bold mt-1">
                        {indicator.value} 
                        {indicator.expected && (
                          <span className="text-sm text-muted-foreground ml-2">
                            (expected: {indicator.expected})
                          </span>
                        )}
                      </p>
                    </div>
                    <Badge variant={indicator.status === "alert" ? "destructive" : "secondary"}>
                      {indicator.status.toUpperCase()}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 