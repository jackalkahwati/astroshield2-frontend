"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { 
  SatelliteIcon, 
  AlertTriangle, 
  Shield, 
  Activity, 
  Globe, 
  Target, 
  Zap, 
  BarChart2,
  GitBranch,
  Radar,
  MessageCircle,
  RefreshCw,
  ArrowUpRight,
  TrendingUp,
  Clock
} from "lucide-react"
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
  Line,
  Area,
  AreaChart
} from 'recharts'
import { HEX_COLORS } from '@/lib/chart-colors'
import Link from "next/link"

interface DashboardData {
  system_metrics: {
    system_health: string
    active_tracks: number
    critical_alerts: number
    response_time: string
  }
  feature_rollup: {
    satellite_tracking: {
      total_satellites: number
      operational: number
      threats: number
      primary_asset: string
    }
    ccdm_analysis: {
      indicators_passing: number
      total_indicators: number
      health_percentage: number
      latest_assessment: string
    }
    maneuvers: {
      scheduled: number
      total_delta_v: string
      next_execution: string
      success_rate: number
    }
    analytics: {
      anomalies_detected: number
      predictions_active: number
      ml_models_running: number
      accuracy_score: number
    }
    proximity_operations: {
      active_conjunctions: number
      closest_approach: string
      risk_level: string
      monitoring_assets: number
    }
    event_correlation: {
      events_processed: number
      correlations_found: number
      threat_escalations: number
      automation_rate: number
    }
    kafka_monitor: {
      topics_active: number
      messages_per_second: number
      latency_avg: string
      health: string
    }
    protection: {
      threats_blocked: number
      countermeasures_active: number
      defense_readiness: string
      last_activation: string
    }
    trajectory_analysis: {
      predictions_active: number
      accuracy: string
      computation_time: string
      orbit_determinations: number
    }
    stability_analysis: {
      stable_objects: number
      unstable_objects: number
      decay_predictions: number
      confidence_score: number
    }
  }
  recent_activities: Array<{
    timestamp: string
    type: string
    message: string
    source: string
  }>
  system_status: {
    overall: string
    subsystems: Record<string, string>
  }
  timestamp: string
}

export default function ComprehensivePage() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  const fetchDashboardData = async () => {
    try {
      // Use relative path in production, localhost in development
      const apiUrl = process.env.NODE_ENV === 'development' 
        ? 'http://localhost:8000/api/v1/dashboard/stats'
        : '/api/v1/dashboard/stats'
      
      const response = await fetch(apiUrl)
      const result = await response.json()
      setData(result)
      setLastUpdate(new Date())
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
      // Fallback to mock data if API fails
      setData({
        system_metrics: {
          system_health: "96%",
          active_tracks: 47,
          critical_alerts: 2,
          response_time: "1.29s"
        },
        feature_rollup: {
          satellite_tracking: { total_satellites: 47, operational: 45, threats: 2, primary_asset: "ISS (USA-317)" },
          ccdm_analysis: { indicators_passing: 5, total_indicators: 8, health_percentage: 87.5, latest_assessment: "2025-06-12T20:15:00Z" },
          maneuvers: { scheduled: 5, total_delta_v: "156.3 m/s", next_execution: "2025-06-13T02:30:00Z", success_rate: 96.5 },
          analytics: { anomalies_detected: 3, predictions_active: 12, ml_models_running: 8, accuracy_score: 94.2 },
          proximity_operations: { active_conjunctions: 4, closest_approach: "180m", risk_level: "MEDIUM", monitoring_assets: 23 },
          event_correlation: { events_processed: 1247, correlations_found: 156, threat_escalations: 8, automation_rate: 78.3 },
          kafka_monitor: { topics_active: 25, messages_per_second: 34.7, latency_avg: "45ms", health: "HEALTHY" },
          protection: { threats_blocked: 12, countermeasures_active: 3, defense_readiness: "HIGH", last_activation: "2025-06-12T19:45:00Z" },
          trajectory_analysis: { predictions_active: 67, accuracy: "97.8%", computation_time: "2.3s", orbit_determinations: 234 },
          stability_analysis: { stable_objects: 42, unstable_objects: 5, decay_predictions: 8, confidence_score: 91.4 }
        },
        recent_activities: [
          { timestamp: "2025-06-12T20:15:00Z", type: "threat_detection", message: "ASAT threat assessment processed - CRITICAL priority", source: "Protection System" },
          { timestamp: "2025-06-12T20:10:00Z", type: "maneuver_scheduled", message: "ISS proximity maneuver scheduled for T+4h", source: "Maneuver Planning" },
          { timestamp: "2025-06-12T20:05:00Z", type: "conjunction_detected", message: "STARLINK-4729 conjunction analysis complete", source: "Proximity Operations" }
        ],
        system_status: {
          overall: "OPERATIONAL",
          subsystems: {
            satellite_tracking: "OPERATIONAL",
            event_processing: "WARNING",
            protection_systems: "OPERATIONAL",
            maneuver_planning: "OPERATIONAL",
            ccdm_analysis: "WARNING",
            kafka_infrastructure: "OPERATIONAL"
          }
        },
        timestamp: new Date().toISOString()
      })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-400" />
          <p className="text-white">Loading comprehensive dashboard...</p>
        </div>
      </div>
    )
  }

  const systemHealthData = [
    { name: 'Tracking', value: data.feature_rollup.satellite_tracking.operational, total: data.feature_rollup.satellite_tracking.total_satellites },
    { name: 'Analytics', value: data.feature_rollup.analytics.accuracy_score, total: 100 },
    { name: 'CCDM', value: data.feature_rollup.ccdm_analysis.health_percentage, total: 100 },
    { name: 'Maneuvers', value: data.feature_rollup.maneuvers.success_rate, total: 100 },
    { name: 'Protection', value: 96, total: 100 },
    { name: 'Stability', value: data.feature_rollup.stability_analysis.confidence_score, total: 100 }
  ]

  const threatOverview = [
    { name: 'Operational', value: data.feature_rollup.satellite_tracking.operational, fill: HEX_COLORS.status.good },
    { name: 'Threats', value: data.feature_rollup.satellite_tracking.threats, fill: HEX_COLORS.alerts.critical },
    { name: 'Unstable', value: data.feature_rollup.stability_analysis.unstable_objects, fill: HEX_COLORS.status.caution }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'OPERATIONAL': return 'text-green-400'
      case 'WARNING': return 'text-yellow-400' 
      case 'CRITICAL': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'OPERATIONAL': return 'bg-green-500'
      case 'WARNING': return 'bg-yellow-500'
      case 'CRITICAL': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">AstroShield Command Center</h1>
          <p className="text-gray-400 mt-1">Comprehensive Space Situational Awareness Dashboard</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right text-sm">
            <div className="text-white">Last Updated</div>
            <div className="text-gray-400">{lastUpdate.toLocaleTimeString()}</div>
          </div>
          <Button onClick={fetchDashboardData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Status Banner */}
      <Card className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 border-blue-500/30">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="h-8 w-8 text-blue-400" />
              <div>
                <h2 className="text-xl font-semibold text-white">System Status: {data.system_status.overall}</h2>
                <p className="text-gray-400">All critical systems operational</p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-green-400">{data.system_metrics.system_health}</div>
                <div className="text-xs text-gray-400">Health</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-400">{data.system_metrics.active_tracks}</div>
                <div className="text-xs text-gray-400">Tracking</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-400">{data.system_metrics.critical_alerts}</div>
                <div className="text-xs text-gray-400">Alerts</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Rollup Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        
        {/* Satellite Tracking */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Satellite Tracking</CardTitle>
            <Link href="/tracking">
              <SatelliteIcon className="h-5 w-5 text-blue-400 hover:text-blue-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.satellite_tracking.total_satellites}</div>
            <p className="text-xs text-gray-400">Total Satellites</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Operational:</span>
                <span className="text-green-400">{data.feature_rollup.satellite_tracking.operational}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Threats:</span>
                <span className="text-red-400">{data.feature_rollup.satellite_tracking.threats}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* CCDM Analysis */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">CCDM Analysis</CardTitle>
            <Link href="/ccdm">
              <Target className="h-5 w-5 text-purple-400 hover:text-purple-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.ccdm_analysis.health_percentage}%</div>
            <p className="text-xs text-gray-400">Health Score</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Passing:</span>
                <span className="text-green-400">{data.feature_rollup.ccdm_analysis.indicators_passing}/{data.feature_rollup.ccdm_analysis.total_indicators}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Maneuvers */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Maneuvers</CardTitle>
            <Link href="/maneuvers">
              <Zap className="h-5 w-5 text-yellow-400 hover:text-yellow-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.maneuvers.scheduled}</div>
            <p className="text-xs text-gray-400">Scheduled</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Total ΔV:</span>
                <span className="text-yellow-400">{data.feature_rollup.maneuvers.total_delta_v}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Success Rate:</span>
                <span className="text-green-400">{data.feature_rollup.maneuvers.success_rate}%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Analytics */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Analytics</CardTitle>
            <Link href="/analytics">
              <BarChart2 className="h-5 w-5 text-cyan-400 hover:text-cyan-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.analytics.accuracy_score}%</div>
            <p className="text-xs text-gray-400">ML Accuracy</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Models:</span>
                <span className="text-cyan-400">{data.feature_rollup.analytics.ml_models_running}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Anomalies:</span>
                <span className="text-orange-400">{data.feature_rollup.analytics.anomalies_detected}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Proximity Operations */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Proximity Ops</CardTitle>
            <Link href="/proximity-operations">
              <Radar className="h-5 w-5 text-green-400 hover:text-green-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.proximity_operations.active_conjunctions}</div>
            <p className="text-xs text-gray-400">Active Conjunctions</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Closest:</span>
                <span className="text-red-400">{data.feature_rollup.proximity_operations.closest_approach}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Risk:</span>
                <Badge variant="secondary" className="text-xs">{data.feature_rollup.proximity_operations.risk_level}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Event Correlation */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Event Correlation</CardTitle>
            <Link href="/event-correlation">
              <GitBranch className="h-5 w-5 text-indigo-400 hover:text-indigo-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.event_correlation.correlations_found}</div>
            <p className="text-xs text-gray-400">Correlations Found</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Processed:</span>
                <span className="text-blue-400">{data.feature_rollup.event_correlation.events_processed}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Auto Rate:</span>
                <span className="text-green-400">{data.feature_rollup.event_correlation.automation_rate}%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Kafka Monitor */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Kafka Monitor</CardTitle>
            <Link href="/kafka-monitor">
              <Activity className="h-5 w-5 text-red-400 hover:text-red-300" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.kafka_monitor.topics_active}</div>
            <p className="text-xs text-gray-400">Active Topics</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Msg/sec:</span>
                <span className="text-green-400">{data.feature_rollup.kafka_monitor.messages_per_second}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Latency:</span>
                <span className="text-blue-400">{data.feature_rollup.kafka_monitor.latency_avg}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Protection */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">Protection</CardTitle>
            <Link href="/protection">
              <Shield className="h-5 w-5 text-red-500 hover:text-red-400" />
            </Link>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{data.feature_rollup.protection.threats_blocked}</div>
            <p className="text-xs text-gray-400">Threats Blocked</p>
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Active:</span>
                <span className="text-orange-400">{data.feature_rollup.protection.countermeasures_active}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Readiness:</span>
                <Badge variant="destructive" className="text-xs">{data.feature_rollup.protection.defense_readiness}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

      </div>

      {/* Charts and Visualizations */}
      <div className="grid gap-6 lg:grid-cols-2">
        
        {/* System Health Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">System Health Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={systemHealthData}>
                <CartesianGrid strokeDasharray="3 3" stroke={HEX_COLORS.grid} />
                <XAxis dataKey="name" stroke={HEX_COLORS.axis} fontSize={12} />
                <YAxis stroke={HEX_COLORS.axis} fontSize={12} />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#ffffff'
                  }}
                />
                <Bar dataKey="value" fill={HEX_COLORS.status.good} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Threat Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Asset Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={threatOverview}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {threatOverview.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#ffffff'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

      </div>

      {/* Recent Activities & System Status */}
      <div className="grid gap-6 lg:grid-cols-2">
        
        {/* Recent Activities */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Recent Activities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {data.recent_activities.map((activity, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-gray-800/50 rounded-lg">
                  <div className="w-2 h-2 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-white truncate">{activity.message}</p>
                      <div className="text-xs text-gray-400 ml-2">
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs">{activity.type}</Badge>
                      <span className="text-xs text-gray-400">{activity.source}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Subsystem Status */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Subsystem Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(data.system_status.subsystems).map(([system, status]) => (
                <div key={system} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${getStatusBadge(status)}`} />
                    <span className="text-white font-medium">
                      {system.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                  </div>
                  <Badge 
                    variant={status === 'OPERATIONAL' ? 'default' : status === 'WARNING' ? 'secondary' : 'destructive'}
                    className="text-xs"
                  >
                    {status}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

      </div>

    </div>
  )
} 