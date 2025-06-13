"use client"

import React, { useState, useEffect, useCallback, useRef } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  AlertTriangle, 
  Shield, 
  Target,
  Activity,
  Gauge,
  Settings,
  Zap,
  TrendingUp,
  TrendingDown,
  Circle,
  Square,
  Triangle,
  Crosshair,
  Navigation,
  Satellite,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  RefreshCw
} from 'lucide-react'
import { format } from 'date-fns'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

// Dynamic 3D visualization
const Proximity3DView = dynamic(() => import('./proximity-3d-view'), { 
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">Loading 3D visualization...</div>
})

// Utility function for safe date formatting
const formatDateSafe = (dateString: string | null | undefined, formatStr: string, fallback: string = 'Invalid time'): string => {
  try {
    if (!dateString) return fallback
    const date = new Date(dateString)
    if (isNaN(date.getTime())) return fallback
    return format(date, formatStr)
  } catch (error) {
    console.error('Date formatting error:', error)
    return fallback
  }
}

interface ProximityEvent {
  event_id: string
  primary_object: string
  secondary_object: string
  proximity_type: string
  threat_level: string
  range_km: number
  relative_velocity_km_s: number
  time_to_closest_approach: string
  collision_probability: number
  event_entered: boolean
  threshold_scores: Record<string, number>
  metadata: Record<string, any>
}

interface RiskToleranceProfile {
  asset_id: string
  asset_value: string
  mission_phase: string
  threat_environment: string
  collision_risk_tolerance: number
  proximity_risk_tolerance: number
  maneuver_risk_tolerance: number
  fuel_remaining_percent: number
  operational_constraints: Record<string, any>
}

interface ProximityThreshold {
  name: string
  range_km: number
  relative_velocity_km_s: number
  time_to_closest_approach_hours: number
  probability_threshold: number
  active: boolean
}

interface ExitCondition {
  event_id: string
  condition_type: string
  progress: number
  estimated_completion: string
  status: 'monitoring' | 'approaching' | 'met'
}

export function ProximityOperationsCenter() {
  // Mock data for demo
  const mockProximityEvents: ProximityEvent[] = [
    {
      event_id: 'PROX-20250609-001',
      primary_object: 'SAT-12345',
      secondary_object: 'DEB-67890',
      proximity_type: 'natural_conjunction',
      threat_level: 'high',
      range_km: 2.5,
      relative_velocity_km_s: 0.15,
      time_to_closest_approach: new Date(Date.now() + 3600000).toISOString(),
      collision_probability: 0.0001,
      event_entered: true,
      threshold_scores: { critical: 0.8, high: 0.9, moderate: 1.0, low: 1.0 },
      metadata: { minimum_range_km: 1.8, miss_distance: 2.1 }
    },
    {
      event_id: 'PROX-20250609-002',
      primary_object: 'SAT-98765',
      secondary_object: 'SAT-11111',
      proximity_type: 'controlled_approach',
      threat_level: 'moderate',
      range_km: 15.2,
      relative_velocity_km_s: 0.08,
      time_to_closest_approach: new Date(Date.now() + 7200000).toISOString(),
      collision_probability: 0.00001,
      event_entered: true,
      threshold_scores: { critical: 0.2, high: 0.4, moderate: 0.8, low: 1.0 },
      metadata: { minimum_range_km: 12.5, approach_type: 'RPO' }
    }
  ]

  const mockRiskProfiles: Record<string, RiskToleranceProfile> = {
    'SAT-12345': {
      asset_id: 'SAT-12345',
      asset_value: 'critical',
      mission_phase: 'operational',
      threat_environment: 'high',
      collision_risk_tolerance: 0.15,
      proximity_risk_tolerance: 0.25,
      maneuver_risk_tolerance: 0.35,
      fuel_remaining_percent: 78.5,
      operational_constraints: { maneuver_window: '06:00-18:00 UTC', max_delta_v: 5.0 }
    },
    'SAT-98765': {
      asset_id: 'SAT-98765',
      asset_value: 'high',
      mission_phase: 'maneuvering',
      threat_environment: 'moderate',
      collision_risk_tolerance: 0.08,
      proximity_risk_tolerance: 0.12,
      maneuver_risk_tolerance: 0.20,
      fuel_remaining_percent: 45.2,
      operational_constraints: { maneuver_window: 'anytime', max_delta_v: 2.5 }
    }
  }

  const mockExitConditions: ExitCondition[] = [
    {
      event_id: 'PROX-20250609-001',
      condition_type: 'range_threshold',
      progress: 65,
      estimated_completion: new Date(Date.now() + 1800000).toISOString(),
      status: 'approaching'
    },
    {
      event_id: 'PROX-20250609-002',
      condition_type: 'time_threshold',
      progress: 25,
      estimated_completion: new Date(Date.now() + 5400000).toISOString(),
      status: 'monitoring'
    }
  ]

  const [activeEvents, setActiveEvents] = useState<ProximityEvent[]>(mockProximityEvents)
  const [selectedEvent, setSelectedEvent] = useState<ProximityEvent | null>(mockProximityEvents[0])
  const [riskProfiles, setRiskProfiles] = useState<Record<string, RiskToleranceProfile>>(mockRiskProfiles)
  const [thresholds, setThresholds] = useState<Record<string, ProximityThreshold>>({
    critical: { name: 'Critical', range_km: 5, relative_velocity_km_s: 0.5, time_to_closest_approach_hours: 1, probability_threshold: 0.001, active: true },
    high: { name: 'High', range_km: 25, relative_velocity_km_s: 0.3, time_to_closest_approach_hours: 6, probability_threshold: 0.0001, active: true },
    moderate: { name: 'Moderate', range_km: 100, relative_velocity_km_s: 0.2, time_to_closest_approach_hours: 24, probability_threshold: 0.00001, active: true },
    low: { name: 'Low', range_km: 200, relative_velocity_km_s: 0.1, time_to_closest_approach_hours: 48, probability_threshold: 0.000001, active: true }
  })
  const [exitConditions, setExitConditions] = useState<ExitCondition[]>(mockExitConditions)
  const [historicalData, setHistoricalData] = useState<any[]>([
    { time: new Date(Date.now() - 3600000).toISOString(), range: 8.5, velocity: 0.12, probability: 0.00008 },
    { time: new Date(Date.now() - 1800000).toISOString(), range: 5.2, velocity: 0.14, probability: 0.00012 },
    { time: new Date(Date.now() - 900000).toISOString(), range: 3.1, velocity: 0.15, probability: 0.00015 },
    { time: new Date().toISOString(), range: 2.5, velocity: 0.15, probability: 0.0001 }
  ])
  const [activeTab, setActiveTab] = useState('monitoring')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const wsRef = useRef<WebSocket | null>(null)

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!wsRef.current) {
      wsRef.current = new WebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/proximity')
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        switch (data.type) {
          case 'proximity_event':
            handleProximityEvent(data.payload)
            break
          case 'threshold_breach':
            handleThresholdBreach(data.payload)
            break
          case 'exit_condition_update':
            handleExitConditionUpdate(data.payload)
            break
        }
      }
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Fetch initial data
  useEffect(() => {
    fetchActiveEvents()
    fetchRiskProfiles()
    fetchExitConditions()
    
    // Add fallback data if API returns empty
    setTimeout(() => {
      if (activeEvents.length === 0) {
        setActiveEvents(mockProximityEvents)
        setSelectedEvent(mockProximityEvents[0])
      }
      if (Object.keys(riskProfiles).length === 0) {
        setRiskProfiles(mockRiskProfiles)
      }
      if (exitConditions.length === 0) {
        setExitConditions(mockExitConditions)
      }
    }, 2000)
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchActiveEvents()
        fetchExitConditions()
      }, 5000) // Every 5 seconds
      
      return () => clearInterval(interval)
    }
  }, [autoRefresh, activeEvents.length, riskProfiles, exitConditions.length, mockProximityEvents, mockRiskProfiles, mockExitConditions])

  const fetchActiveEvents = async () => {
    try {
      const response = await fetch('/api/v1/proximity/active-events')
      if (response.ok) {
        const data = await response.json()
        // Use API data if available, otherwise fallback to mock data
        setActiveEvents(data.length > 0 ? data : mockProximityEvents)
        if (data.length > 0 && !selectedEvent) {
          setSelectedEvent(data[0])
        }
      }
    } catch (error) {
      console.error('Failed to fetch active events:', error)
      // On error, use mock data as fallback
      setActiveEvents(mockProximityEvents)
      if (!selectedEvent) {
        setSelectedEvent(mockProximityEvents[0])
      }
    }
  }

  const fetchRiskProfiles = async () => {
    try {
      const response = await fetch('/api/v1/risk-tolerance/profiles')
      if (response.ok) {
        const data = await response.json()
        if (data.length > 0) {
          const profileMap = data.reduce((acc: any, profile: RiskToleranceProfile) => {
            acc[profile.asset_id] = profile
            return acc
          }, {})
          setRiskProfiles(profileMap)
        } else {
          // Use mock data as fallback when API returns empty
          setRiskProfiles(mockRiskProfiles)
        }
      }
    } catch (error) {
      console.error('Failed to fetch risk profiles:', error)
      // On error, use mock data as fallback
      setRiskProfiles(mockRiskProfiles)
    }
  }

  const fetchExitConditions = async () => {
    try {
      const response = await fetch('/api/v1/proximity/exit-conditions')
      if (response.ok) {
        const data = await response.json()
        // Use API data if available, otherwise fallback to mock data
        setExitConditions(data.length > 0 ? data : mockExitConditions)
      }
    } catch (error) {
      console.error('Failed to fetch exit conditions:', error)
      // On error, use mock data as fallback
      setExitConditions(mockExitConditions)
    }
  }

  const handleProximityEvent = (event: ProximityEvent) => {
    setActiveEvents(prev => {
      const existing = prev.findIndex(e => e.event_id === event.event_id)
      if (existing >= 0) {
        const updated = [...prev]
        updated[existing] = event
        return updated
      }
      return [...prev, event]
    })
    
    // Update historical data
    setHistoricalData(prev => [...prev, {
      time: new Date().toISOString(),
      range: event.range_km,
      velocity: event.relative_velocity_km_s,
      probability: event.collision_probability
    }].slice(-100)) // Keep last 100 points
  }

  const handleThresholdBreach = (breach: any) => {
    // Show alert for threshold breach
    console.log('Threshold breach:', breach)
  }

  const handleExitConditionUpdate = (update: any) => {
    setExitConditions(prev => {
      const existing = prev.findIndex(e => e.event_id === update.event_id)
      if (existing >= 0) {
        const updated = [...prev]
        updated[existing] = { ...updated[existing], ...update }
        return updated
      }
      return prev
    })
  }

  const updateRiskTolerance = async (assetId: string, field: string, value: number) => {
    try {
      const response = await fetch(`/api/v1/risk-tolerance/profiles/${assetId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value })
      })
      
      if (response.ok) {
        const updated = await response.json()
        setRiskProfiles(prev => ({
          ...prev,
          [assetId]: updated
        }))
      }
    } catch (error) {
      console.error('Failed to update risk tolerance:', error)
    }
  }

  const updateThreshold = (name: string, field: string, value: number) => {
    setThresholds(prev => ({
      ...prev,
      [name]: {
        ...prev[name],
        [field]: value
      }
    }))
  }

  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-100'
      case 'high': return 'text-orange-600 bg-orange-100'
      case 'moderate': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-blue-600 bg-blue-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getProximityTypeIcon = (type: string) => {
    switch (type) {
      case 'intercept_trajectory': return <Crosshair className="h-4 w-4" />
      case 'controlled_approach': return <Navigation className="h-4 w-4" />
      case 'natural_conjunction': return <Circle className="h-4 w-4" />
      case 'station_keeping': return <Square className="h-4 w-4" />
      default: return <Triangle className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex justify-end items-center">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Switch
              checked={autoRefresh}
              onCheckedChange={setAutoRefresh}
            />
            <span className="text-sm">Auto-refresh</span>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchActiveEvents()
              fetchExitConditions()
            }}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Critical Alerts */}
      {activeEvents.filter(e => e.threat_level === 'critical').length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>{activeEvents.filter(e => e.threat_level === 'critical').length} critical proximity events active!</strong>
            {' '}Immediate attention required for collision avoidance.
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="monitoring">
            <Activity className="h-4 w-4 mr-2" />
            Live Monitoring
          </TabsTrigger>
          <TabsTrigger value="risk-tolerance">
            <Shield className="h-4 w-4 mr-2" />
            Risk Tolerance
          </TabsTrigger>
          <TabsTrigger value="thresholds">
            <Gauge className="h-4 w-4 mr-2" />
            Thresholds
          </TabsTrigger>
          <TabsTrigger value="exit-conditions">
            <Target className="h-4 w-4 mr-2" />
            Exit Conditions
          </TabsTrigger>
        </TabsList>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Active Events List */}
            <Card>
              <CardHeader>
                <CardTitle>Active Proximity Events</CardTitle>
                <CardDescription>
                  {activeEvents.length} events being monitored
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {activeEvents.map((event) => (
                    <div
                      key={event.event_id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedEvent?.event_id === event.event_id ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => setSelectedEvent(event)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {getProximityTypeIcon(event.proximity_type)}
                          <span className="font-mono text-sm">{event.event_id}</span>
                        </div>
                        <Badge className={getThreatLevelColor(event.threat_level)}>
                          {event.threat_level}
                        </Badge>
                      </div>
                      
                      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500">Range:</span>
                          <span className="ml-1 font-medium">{event.range_km.toFixed(1)} km</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Velocity:</span>
                          <span className="ml-1 font-medium">{event.relative_velocity_km_s.toFixed(3)} km/s</span>
                        </div>
                        <div>
                          <span className="text-gray-500">P(collision):</span>
                          <span className="ml-1 font-medium">{event.collision_probability.toExponential(1)}</span>
                        </div>
                      </div>
                      
                      <div className="mt-2 text-xs text-gray-500">
                        {event.primary_object} ↔ {event.secondary_object}
                      </div>
                    </div>
                  ))}
                  
                  {activeEvents.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-500" />
                      <p>No active proximity events</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Event Details */}
            {selectedEvent && (
              <Card>
                <CardHeader>
                  <CardTitle>Event Details: {selectedEvent.event_id}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* 3D Visualization */}
                  <div className="h-64">
                    <Proximity3DView event={selectedEvent} />
                  </div>
                  
                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Time to Closest Approach</p>
                      <p className="text-lg font-semibold">
                        {formatDateSafe(selectedEvent.time_to_closest_approach, 'HH:mm:ss', 'Unknown')}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Minimum Range</p>
                      <p className="text-lg font-semibold">
                        {selectedEvent.metadata?.minimum_range_km?.toFixed(2) || selectedEvent.range_km.toFixed(2)} km
                      </p>
                    </div>
                  </div>
                  
                  {/* Threshold Scores */}
                  <div>
                    <p className="text-sm font-medium mb-2">Threshold Scores</p>
                    {Object.entries(selectedEvent.threshold_scores).map(([name, score]) => (
                      <div key={name} className="flex items-center justify-between mb-2">
                        <span className="text-sm capitalize">{name}</span>
                        <div className="flex items-center gap-2">
                          <Progress value={score * 100} className="w-24 h-2" />
                          <span className="text-xs text-gray-500">{(score * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Real-time Charts */}
          <Card>
            <CardHeader>
              <CardTitle>Real-time Proximity Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                                              tickFormatter={(time) => formatDateSafe(time, 'HH:mm:ss', 'N/A')}
                    />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip 
                                              labelFormatter={(time) => formatDateSafe(time, 'HH:mm:ss', 'N/A')}
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="range" 
                      stroke="#3b82f6" 
                      name="Range (km)"
                      strokeWidth={2}
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="velocity" 
                      stroke="#ef4444" 
                      name="Velocity (km/s)"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk-tolerance" className="space-y-4">
          {Object.entries(riskProfiles).map(([assetId, profile]) => (
            <Card key={assetId}>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle>{assetId}</CardTitle>
                    <CardDescription>
                      {profile.asset_value} value • {profile.mission_phase} phase • {profile.threat_environment} environment
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Gauge className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-500">
                      Fuel: {profile.fuel_remaining_percent.toFixed(0)}%
                    </span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Collision Risk Tolerance */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Collision Risk Tolerance</span>
                    <span className="text-sm text-gray-500">
                      {(profile.collision_risk_tolerance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    value={[profile.collision_risk_tolerance * 100]}
                    onValueChange={([value]) => updateRiskTolerance(assetId, 'collision_risk_tolerance', value / 100)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Proximity Risk Tolerance */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Proximity Risk Tolerance</span>
                    <span className="text-sm text-gray-500">
                      {(profile.proximity_risk_tolerance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    value={[profile.proximity_risk_tolerance * 100]}
                    onValueChange={([value]) => updateRiskTolerance(assetId, 'proximity_risk_tolerance', value / 100)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Maneuver Risk Tolerance */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Maneuver Risk Tolerance</span>
                    <span className="text-sm text-gray-500">
                      {(profile.maneuver_risk_tolerance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    value={[profile.maneuver_risk_tolerance * 100]}
                    onValueChange={([value]) => updateRiskTolerance(assetId, 'maneuver_risk_tolerance', value / 100)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Operational Constraints */}
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm font-medium text-gray-700 mb-2">Operational Constraints</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(profile.operational_constraints).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-gray-500">{key.replace('_', ' ')}:</span>
                        <span className="ml-1 font-medium">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Maneuver Risk Tolerance */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Maneuver Risk Tolerance</span>
                    <span className="text-sm text-gray-500">
                      {(profile.maneuver_risk_tolerance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider
                    value={[profile.maneuver_risk_tolerance * 100]}
                    onValueChange={([value]) => updateRiskTolerance(assetId, 'maneuver_risk_tolerance', value / 100)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Operational Constraints */}
                {Object.keys(profile.operational_constraints).length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Operational Constraints</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(profile.operational_constraints).map(([key, value]) => (
                        <Badge key={key} variant="outline" className="text-xs">
                          {key}: {String(value)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="thresholds" className="space-y-4">
          {Object.entries(thresholds).map(([key, threshold]) => (
            <Card key={key}>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle>{threshold.name} Threshold</CardTitle>
                  <Switch
                    checked={threshold.active}
                    onCheckedChange={(checked) => updateThreshold(key, 'active', checked ? 1 : 0)}
                  />
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Range Threshold */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Range Threshold</span>
                    <span className="text-sm text-gray-500">{threshold.range_km} km</span>
                  </div>
                  <Slider
                    value={[threshold.range_km]}
                    onValueChange={([value]) => updateThreshold(key, 'range_km', value)}
                    max={500}
                    step={1}
                    className="w-full"
                    disabled={!threshold.active}
                  />
                </div>

                {/* Velocity Threshold */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Relative Velocity Threshold</span>
                    <span className="text-sm text-gray-500">{threshold.relative_velocity_km_s} km/s</span>
                  </div>
                  <Slider
                    value={[threshold.relative_velocity_km_s * 100]}
                    onValueChange={([value]) => updateThreshold(key, 'relative_velocity_km_s', value / 100)}
                    max={100}
                    step={1}
                    className="w-full"
                    disabled={!threshold.active}
                  />
                </div>

                {/* TCA Threshold */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Time to Closest Approach</span>
                    <span className="text-sm text-gray-500">{threshold.time_to_closest_approach_hours} hours</span>
                  </div>
                  <Slider
                    value={[threshold.time_to_closest_approach_hours]}
                    onValueChange={([value]) => updateThreshold(key, 'time_to_closest_approach_hours', value)}
                    max={72}
                    step={1}
                    className="w-full"
                    disabled={!threshold.active}
                  />
                </div>

                {/* Probability Threshold */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Collision Probability Threshold</span>
                    <span className="text-sm text-gray-500">{threshold.probability_threshold.toExponential(1)}</span>
                  </div>
                  <Slider
                    value={[Math.log10(threshold.probability_threshold) + 6]}
                    onValueChange={([value]) => updateThreshold(key, 'probability_threshold', Math.pow(10, value - 6))}
                    max={3}
                    min={0}
                    step={0.1}
                    className="w-full"
                    disabled={!threshold.active}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="exit-conditions">
          <Card>
            <CardHeader>
              <CardTitle>Active Event Exit Conditions</CardTitle>
              <CardDescription>
                Monitoring conditions for event closure
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {exitConditions.map((condition) => (
                  <div key={condition.event_id} className="p-3 border rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-sm">{condition.event_id}</span>
                      <Badge 
                        variant={condition.status === 'met' ? 'default' : 'outline'}
                        className={
                          condition.status === 'met' ? 'bg-green-100 text-green-800' :
                          condition.status === 'approaching' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }
                      >
                        {condition.status}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">{condition.condition_type}</span>
                        <span className="text-gray-500">
                          Est. {formatDateSafe(condition.estimated_completion, 'HH:mm', 'Unknown')}
                        </span>
                      </div>
                      
                      <Progress value={condition.progress} className="h-2" />
                      
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>Progress: {condition.progress.toFixed(0)}%</span>
                        {condition.status === 'approaching' && (
                          <span className="text-yellow-600">
                            <AlertCircle className="h-3 w-3 inline mr-1" />
                            Approaching exit
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                
                {exitConditions.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <Info className="h-12 w-12 mx-auto mb-2" />
                    <p>No active exit conditions</p>
                    <p className="text-sm">Exit conditions appear when proximity events are active</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 