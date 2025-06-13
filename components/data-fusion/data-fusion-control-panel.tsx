"use client"

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  Layers, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Zap,
  Users,
  BarChart3,
  LineChart,
  PieChart,
  Clock,
  Gauge,
  Filter,
  Download,
  RefreshCw,
  Info,
  Star,
  Award
} from 'lucide-react'
import { format } from 'date-fns'
import { 
  BarChart, 
  Bar, 
  LineChart as RechartsLineChart, 
  Line, 
  PieChart as RechartsPieChart, 
  Pie, 
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts'

interface VendorProfile {
  vendor_id: string
  reliability: string
  total_predictions: number
  success_rate: number
  average_latency: number
  availability: number
  strengths: string[]
}

interface FusionStatistics {
  total_fusions: number
  by_type: Record<string, number>
  average_vendors: number
  average_agreement: number
  active_buffers: number
  pending_predictions: number
  vendor_count: number
  vendor_reliability: Record<string, number>
}

interface FusedPrediction {
  fusion_id: string
  prediction_type: string
  timestamp: string
  confidence: number
  vendor_count: number
  vendor_ids: string[]
  agreement_score: number
  reliability_score: number
  fusion_method: string
  fusion_rationale: string[]
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

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

export function DataFusionControlPanel() {
  const [vendorProfiles, setVendorProfiles] = useState<Record<string, VendorProfile>>({})
  const [fusionStats, setFusionStats] = useState<FusionStatistics | null>(null)
  const [recentFusions, setRecentFusions] = useState<FusedPrediction[]>([])
  const [selectedVendor, setSelectedVendor] = useState<string | null>(null)
  const [selectedPredictionType, setSelectedPredictionType] = useState<string>('all')
  const [timeRange, setTimeRange] = useState<string>('24h')
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(false)

  // Fetch data
  useEffect(() => {
    fetchVendorPerformance()
    fetchFusionStatistics()
    fetchRecentFusions()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchFusionStatistics()
      fetchRecentFusions()
    }, 30000)
    
    return () => clearInterval(interval)
  }, [timeRange, selectedPredictionType])

  const fetchVendorPerformance = async () => {
    try {
      const response = await fetch('/api/v1/data-fusion/vendor-performance')
      if (response.ok) {
        const data = await response.json()
        setVendorProfiles(data.vendors)
      }
    } catch (error) {
      console.error('Failed to fetch vendor performance:', error)
    }
  }

  const fetchFusionStatistics = async () => {
    try {
      const response = await fetch('/api/v1/data-fusion/statistics')
      if (response.ok) {
        const data = await response.json()
        setFusionStats(data)
      }
    } catch (error) {
      console.error('Failed to fetch fusion statistics:', error)
    }
  }

  const fetchRecentFusions = async () => {
    try {
      const params = new URLSearchParams({
        time_range: timeRange,
        type: selectedPredictionType !== 'all' ? selectedPredictionType : ''
      })
      
      const response = await fetch(`/api/v1/data-fusion/recent?${params}`)
      if (response.ok) {
        const data = await response.json()
        setRecentFusions(data)
      }
    } catch (error) {
      console.error('Failed to fetch recent fusions:', error)
    }
  }

  const getReliabilityColor = (reliability: string) => {
    switch (reliability) {
      case 'excellent': return 'text-green-600 bg-green-100'
      case 'good': return 'text-blue-600 bg-blue-100'
      case 'fair': return 'text-yellow-600 bg-yellow-100'
      case 'poor': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getReliabilityIcon = (reliability: string) => {
    switch (reliability) {
      case 'excellent': return <Star className="h-4 w-4" />
      case 'good': return <CheckCircle className="h-4 w-4" />
      case 'fair': return <AlertTriangle className="h-4 w-4" />
      case 'poor': return <XCircle className="h-4 w-4" />
      default: return <Info className="h-4 w-4" />
    }
  }

  // Prepare chart data
  const vendorComparisonData = Object.values(vendorProfiles).map(vendor => ({
    name: vendor.vendor_id,
    success_rate: vendor.success_rate * 100,
    latency: vendor.average_latency,
    availability: vendor.availability * 100,
    predictions: vendor.total_predictions
  }))

  const fusionTypeData = fusionStats ? Object.entries(fusionStats.by_type).map(([type, count]) => ({
    name: type,
    value: count
  })) : []

  const reliabilityDistribution = fusionStats ? Object.entries(fusionStats.vendor_reliability).map(([level, count]) => ({
    name: level,
    value: count
  })) : []

  const vendorRadarData = selectedVendor && vendorProfiles[selectedVendor] ? [{
    metric: 'Success Rate',
    value: vendorProfiles[selectedVendor].success_rate * 100,
    fullMark: 100
  }, {
    metric: 'Low Latency',
    value: Math.max(0, 100 - vendorProfiles[selectedVendor].average_latency),
    fullMark: 100
  }, {
    metric: 'Availability',
    value: vendorProfiles[selectedVendor].availability * 100,
    fullMark: 100
  }, {
    metric: 'Experience',
    value: Math.min(100, vendorProfiles[selectedVendor].total_predictions / 10),
    fullMark: 100
  }] : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Data Fusion Control Panel</h2>
          <p className="text-gray-600">Monitor vendor performance and fusion quality</p>
        </div>
        
        <div className="flex gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchFusionStatistics()
              fetchRecentFusions()
            }}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
          
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      {fusionStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Fusions</p>
                  <p className="text-2xl font-bold">{fusionStats.total_fusions.toLocaleString()}</p>
                </div>
                <Layers className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Avg Agreement</p>
                  <p className="text-2xl font-bold">{(fusionStats.average_agreement * 100).toFixed(1)}%</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Active Vendors</p>
                  <p className="text-2xl font-bold">{fusionStats.vendor_count}</p>
                </div>
                <Users className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Pending</p>
                  <p className="text-2xl font-bold">{fusionStats.pending_predictions}</p>
                </div>
                <Clock className="h-8 w-8 text-orange-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">
            <BarChart3 className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="vendors">
            <Users className="h-4 w-4 mr-2" />
            Vendors
          </TabsTrigger>
          <TabsTrigger value="fusions">
            <Layers className="h-4 w-4 mr-2" />
            Recent Fusions
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <LineChart className="h-4 w-4 mr-2" />
            Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Fusion Types Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Fusion Types Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={fusionTypeData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {fusionTypeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Vendor Reliability Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Vendor Reliability Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={reliabilityDistribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Vendor Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Vendor Performance Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={vendorComparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="success_rate" fill="#10b981" name="Success Rate %" />
                    <Bar dataKey="availability" fill="#3b82f6" name="Availability %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="vendors" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Vendor List */}
            <Card>
              <CardHeader>
                <CardTitle>Vendor Profiles</CardTitle>
                <CardDescription>
                  Click on a vendor to see detailed performance metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(vendorProfiles).map(([vendorId, profile]) => (
                    <div
                      key={vendorId}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedVendor === vendorId ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => setSelectedVendor(vendorId)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{vendorId}</span>
                          <Badge className={getReliabilityColor(profile.reliability)}>
                            <span className="flex items-center gap-1">
                              {getReliabilityIcon(profile.reliability)}
                              {profile.reliability}
                            </span>
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-500">
                          {profile.total_predictions} predictions
                        </div>
                      </div>
                      
                      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-gray-500">Success:</span>
                          <span className="ml-1 font-medium">{(profile.success_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Latency:</span>
                          <span className="ml-1 font-medium">{profile.average_latency.toFixed(0)}ms</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Uptime:</span>
                          <span className="ml-1 font-medium">{(profile.availability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      
                      {profile.strengths.length > 0 && (
                        <div className="mt-2 flex gap-1">
                          {profile.strengths.map(strength => (
                            <Badge key={strength} variant="outline" className="text-xs">
                              {strength}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Vendor Details */}
            {selectedVendor && vendorProfiles[selectedVendor] && (
              <Card>
                <CardHeader>
                  <CardTitle>{selectedVendor} Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={vendorRadarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} />
                        <Radar
                          name={selectedVendor}
                          dataKey="value"
                          stroke="#3b82f6"
                          fill="#3b82f6"
                          fillOpacity={0.6}
                        />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500">Total Predictions</span>
                      <span className="font-medium">{vendorProfiles[selectedVendor].total_predictions}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500">Success Rate</span>
                      <span className="font-medium">{(vendorProfiles[selectedVendor].success_rate * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500">Average Latency</span>
                      <span className="font-medium">{vendorProfiles[selectedVendor].average_latency.toFixed(1)}ms</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500">Availability</span>
                      <span className="font-medium">{(vendorProfiles[selectedVendor].availability * 100).toFixed(2)}%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="fusions" className="space-y-4">
          {/* Filter */}
          <Card>
            <CardContent className="p-4">
              <div className="flex gap-2">
                <Select value={selectedPredictionType} onValueChange={setSelectedPredictionType}>
                  <SelectTrigger className="w-48">
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="pez_wez">PEZ/WEZ</SelectItem>
                    <SelectItem value="ephemeris">Ephemeris</SelectItem>
                    <SelectItem value="maneuver">Maneuver</SelectItem>
                    <SelectItem value="intent">Intent</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Recent Fusions List */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Fusions</CardTitle>
              <CardDescription>
                Latest {recentFusions.length} fused predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentFusions.map((fusion) => (
                  <div key={fusion.fusion_id} className="p-3 border rounded-lg">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm">{fusion.fusion_id}</span>
                          <Badge variant="outline">{fusion.prediction_type}</Badge>
                          <Badge variant="outline">{fusion.fusion_method}</Badge>
                        </div>
                        
                        <div className="mt-1 text-xs text-gray-500">
                          {formatDateSafe(fusion.timestamp, 'PPpp', 'Invalid date')}
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          Confidence: {(fusion.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                          Agreement: {(fusion.agreement_score * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-2 flex items-center gap-4 text-xs">
                      <span className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        {fusion.vendor_count} vendors
                      </span>
                      <span className="flex items-center gap-1">
                        <Gauge className="h-3 w-3" />
                        Reliability: {(fusion.reliability_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="mt-2">
                      <p className="text-xs text-gray-600">{fusion.fusion_rationale[0]}</p>
                    </div>
                    
                    <div className="mt-2 flex gap-1">
                      {fusion.vendor_ids.map(vendorId => (
                        <Badge key={vendorId} variant="outline" className="text-xs">
                          {vendorId}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
                
                {recentFusions.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <Info className="h-12 w-12 mx-auto mb-2" />
                    <p>No recent fusions found</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          {/* Fusion Trends */}
          <Card>
            <CardHeader>
              <CardTitle>Fusion Activity Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsLineChart data={[
                    { time: '00:00', fusions: 45, agreement: 85 },
                    { time: '04:00', fusions: 38, agreement: 82 },
                    { time: '08:00', fusions: 52, agreement: 88 },
                    { time: '12:00', fusions: 61, agreement: 90 },
                    { time: '16:00', fusions: 55, agreement: 87 },
                    { time: '20:00', fusions: 48, agreement: 86 },
                    { time: '24:00', fusions: 42, agreement: 84 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="fusions" 
                      stroke="#3b82f6" 
                      name="Fusions/Hour"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="agreement" 
                      stroke="#10b981" 
                      name="Avg Agreement %"
                    />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Fusion Success Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">98.5%</div>
                <Progress value={98.5} className="mt-2" />
                <p className="text-xs text-gray-500 mt-1">
                  <TrendingUp className="h-3 w-3 inline text-green-500" /> +2.3% from last week
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Average Fusion Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">1.2s</div>
                <Progress value={88} className="mt-2" />
                <p className="text-xs text-gray-500 mt-1">
                  <TrendingDown className="h-3 w-3 inline text-green-500" /> -0.3s from last week
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Data Quality Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">92.3</div>
                <Progress value={92.3} className="mt-2" />
                <p className="text-xs text-gray-500 mt-1">
                  <TrendingUp className="h-3 w-3 inline text-green-500" /> +1.1 from last week
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 