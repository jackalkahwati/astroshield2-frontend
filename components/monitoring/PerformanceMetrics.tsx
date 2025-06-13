"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle2,
  Activity,
  RefreshCw,
  BarChart3
} from "lucide-react"
import { KafkaMetrics, PerformanceMetric } from "@/types/kafka-monitor"

// Mock data for demo/fallback
const mockMetrics: KafkaMetrics = {
  throughput: {
    messagesPerSecond: 1245,
    bytesPerSecond: 15728640, // ~15 MB/s
    peakMessagesPerSecond: 2341,
    peakBytesPerSecond: 31457280 // ~30 MB/s
  },
  latency: {
    average: 45,
    median: 38,
    p95: 127,
    p99: 234
  },
  errors: {
    rate: 0.0023, // 0.23%
    total: 12,
    recentErrors: [
      {
        topic: "ss2.association-message.correlation",
        error: "Message validation failed: invalid correlation ID format",
        timestamp: new Date(Date.now() - 25000).toISOString()
      },
      {
        topic: "ss0.launch-detection.radar",
        error: "Connection timeout to source system RADAR-SITE-2", 
        timestamp: new Date(Date.now() - 67000).toISOString()
      },
      {
        topic: "astroshield.decision-support.recommendations",
        error: "Consumer group rebalancing failed",
        timestamp: new Date(Date.now() - 123000).toISOString()
      }
    ]
  },
  consumers: {
    activeConsumers: 8,
    totalLag: 347,
    consumerGroups: [
      {
        groupId: "astroshield-ss5-processors",
        state: "stable",
        lag: 234
      },
      {
        groupId: "astroshield-event-correlation",
        state: "stable", 
        lag: 89
      },
      {
        groupId: "astroshield-proximity-monitor",
        state: "rebalancing",
        lag: 24
      }
    ]
  },
  producers: {
    activeProducers: 5,
    totalMessagesSent: 45672,
    totalBytesSent: 876543210
  }
}

interface MetricCardProps {
  title: string
  value: string | number
  unit?: string
  trend: number
  trendDirection: 'up' | 'down' | 'stable'
  description: string
  status?: 'good' | 'warning' | 'error'
}

function MetricCard({ title, value, unit, trend, trendDirection, description, status = 'good' }: MetricCardProps) {
  const getTrendIcon = () => {
    switch (trendDirection) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />
      case 'down': return <TrendingDown className="h-4 w-4 text-red-600" />
      default: return <Minus className="h-4 w-4 text-gray-600" />
    }
  }

  const getTrendColor = () => {
    switch (trendDirection) {
      case 'up': return 'text-green-600'
      case 'down': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getStatusColor = () => {
    switch (status) {
      case 'good': return 'border-l-green-500'
      case 'warning': return 'border-l-yellow-500'
      case 'error': return 'border-l-red-500'
      default: return 'border-l-gray-500'
    }
  }

  return (
    <Card className={`border-l-4 ${getStatusColor()}`}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <div className="flex items-baseline space-x-1">
              <p className="text-2xl font-bold">
                {typeof value === 'number' ? value.toLocaleString() : value}
              </p>
              {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
            </div>
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          </div>
          <div className="text-right">
            <div className={`flex items-center space-x-1 ${getTrendColor()}`}>
              {getTrendIcon()}
              <span className="text-sm font-medium">
                {trend > 0 ? '+' : ''}{trend}%
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function PerformanceMetrics() {
  const [metrics, setMetrics] = useState<KafkaMetrics | null>(mockMetrics)
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<string>(new Date().toISOString())
  const [isUsingSampleData, setIsUsingSampleData] = useState(true)

  const fetchMetrics = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/kafka-monitor/metrics/performance')
      if (response.ok) {
        const data = await response.json()
        if (data && Object.keys(data).length > 0) {
          setMetrics(data)
          setIsUsingSampleData(false)
          setLastUpdate(new Date().toISOString())
        } else {
          // Use mock data as fallback when API returns empty
          setMetrics(mockMetrics)
          setIsUsingSampleData(true)
          setLastUpdate(new Date().toISOString())
        }
      }
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error)
      // On error, use mock data as fallback
      setMetrics(mockMetrics)
      setIsUsingSampleData(true)
      setLastUpdate(new Date().toISOString())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
    
    // Add fallback data if API takes too long or returns empty
    setTimeout(() => {
      if (!metrics || (isUsingSampleData && !loading)) {
        setMetrics(mockMetrics)
        setIsUsingSampleData(true)
        setLastUpdate(new Date().toISOString())
      }
    }, 3000)
    
    const interval = setInterval(fetchMetrics, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading && !metrics) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="h-5 w-5 animate-spin mr-2" />
        <span>Loading performance metrics...</span>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Failed to load performance metrics
      </div>
    )
  }

  const getLatencyStatus = (latency: number) => {
    if (latency < 50) return 'good'
    if (latency < 200) return 'warning'
    return 'error'
  }

  const getErrorRateStatus = (rate: number) => {
    if (rate < 0.01) return 'good'
    if (rate < 0.05) return 'warning'
    return 'error'
  }

  const getThroughputStatus = (throughput: number) => {
    if (throughput > 1000) return 'good'
    if (throughput > 100) return 'warning'
    return 'error'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-bold">Performance Metrics</h2>
            {isUsingSampleData && (
              <Badge variant="outline" className="text-xs">
                Sample Data
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground">
            Real-time Kafka performance and health indicators
          </p>
        </div>
        <div className="flex items-center space-x-2 text-sm text-muted-foreground">
          <Activity className="h-4 w-4" />
          <span>{isUsingSampleData ? "Sample data" : "Live data"}</span>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Messages per Second"
          value={metrics.throughput.messagesPerSecond}
          unit="msg/s"
          trend={12}
          trendDirection="up"
          description="Current message throughput"
          status={getThroughputStatus(metrics.throughput.messagesPerSecond)}
        />

        <MetricCard
          title="Average Latency"
          value={metrics.latency.average}
          unit="ms"
          trend={-8}
          trendDirection="down"
          description="End-to-end message latency"
          status={getLatencyStatus(metrics.latency.average)}
        />

        <MetricCard
          title="Error Rate"
          value={(metrics.errors.rate * 100).toFixed(3)}
          unit="%"
          trend={0.01}
          trendDirection="up"
          description="Failed messages percentage"
          status={getErrorRateStatus(metrics.errors.rate)}
        />

        <MetricCard
          title="Consumer Lag"
          value={metrics.consumers.totalLag}
          unit="msgs"
          trend={-15}
          trendDirection="down"
          description="Total consumer lag across topics"
          status={metrics.consumers.totalLag < 1000 ? 'good' : 'warning'}
        />
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Throughput Details */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Throughput Details</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Messages per Second</span>
              <span className="text-sm font-bold">{metrics.throughput.messagesPerSecond}</span>
            </div>
            <Progress value={(metrics.throughput.messagesPerSecond / 5000) * 100} className="h-2" />
            
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Bytes per Second</span>
              <span className="text-sm font-bold">
                {(metrics.throughput.bytesPerSecond / (1024 * 1024)).toFixed(2)} MB/s
              </span>
            </div>
            <Progress value={(metrics.throughput.bytesPerSecond / (50 * 1024 * 1024)) * 100} className="h-2" />
          </CardContent>
        </Card>

        {/* Latency Details */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Latency Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Average</span>
              <span className="text-sm font-bold">{metrics.latency.average}ms</span>
            </div>
            <Progress value={(metrics.latency.average / 500) * 100} className="h-2" />
            
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">95th Percentile</span>
              <span className="text-sm font-bold">{metrics.latency.p95}ms</span>
            </div>
            <Progress value={(metrics.latency.p95 / 1000) * 100} className="h-2" />
            
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">99th Percentile</span>
              <span className="text-sm font-bold">{metrics.latency.p99}ms</span>
            </div>
            <Progress value={(metrics.latency.p99 / 2000) * 100} className="h-2" />
          </CardContent>
        </Card>
      </div>

      {/* Error Analysis and Consumer Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Error Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5" />
              <span>Error Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Error Rate</span>
                <Badge variant={getErrorRateStatus(metrics.errors.rate) === 'good' ? 'default' : 'destructive'}>
                  {(metrics.errors.rate * 100).toFixed(3)}%
                </Badge>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Total Errors</span>
                <span className="text-sm font-bold">{metrics.errors.total}</span>
              </div>

              {metrics.errors.recentErrors.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Recent Errors</h4>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {metrics.errors.recentErrors.slice(0, 3).map((error, index) => (
                      <div key={index} className="text-xs bg-red-50 p-2 rounded border">
                        <div className="flex justify-between">
                          <span className="font-medium">{error.topic}</span>
                          <span className="text-muted-foreground">
                            {new Date(error.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-red-600 truncate">{error.error}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Consumer Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Consumer Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Active Consumers</span>
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-bold">{metrics.consumers.activeConsumers}</span>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Total Consumer Lag</span>
                <div className="flex items-center space-x-2">
                  <Badge variant={metrics.consumers.totalLag < 1000 ? 'default' : 'destructive'}>
                    {metrics.consumers.totalLag} messages
                  </Badge>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Lag Distribution</span>
                  <span className="text-xs text-muted-foreground">
                    {metrics.consumers.totalLag < 1000 ? 'Healthy' : 'Needs Attention'}
                  </span>
                </div>
                <Progress 
                  value={Math.min((metrics.consumers.totalLag / 5000) * 100, 100)} 
                  className="h-2"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Last Update */}
      {lastUpdate && (
        <div className="text-center text-sm text-muted-foreground">
          Last updated: {new Date(lastUpdate).toLocaleString()}
        </div>
      )}
    </div>
  )
} 