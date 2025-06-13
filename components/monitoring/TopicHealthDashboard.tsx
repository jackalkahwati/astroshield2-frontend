"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  Clock,
  Shield,
  Satellite,
  Target,
  Navigation,
  Command,
  Monitor,
  AlertOctagon,
  RefreshCw
} from "lucide-react"
import { TopicHealth, SDATopicCategory } from "@/types/kafka-monitor"

// Mock data for demo/fallback
const mockTopicCategories: SDATopicCategory[] = [
  {
    name: "Data Ingestion (SS0)",
    subsystem: "SS0",
    description: "Launch detection and sensor data ingestion",
    topics: [
      {
        name: "ss0.launch-detection.radar",
        status: "healthy",
        messageCount: 1245,
        consumerLag: 0,
        lastMessage: new Date(Date.now() - 5000).toISOString()
      },
      {
        name: "ss0.sensor.heartbeat",
        status: "healthy", 
        messageCount: 987,
        consumerLag: 2,
        lastMessage: new Date(Date.now() - 3000).toISOString()
      },
      {
        name: "ss0.satellite-tasking.feasibility",
        status: "warning",
        messageCount: 156,
        consumerLag: 15,
        lastMessage: new Date(Date.now() - 45000).toISOString()
      }
    ]
  },
  {
    name: "State Estimation (SS2)",
    subsystem: "SS2", 
    description: "Object tracking and state vector processing",
    topics: [
      {
        name: "ss2.state-vector.full",
        status: "healthy",
        messageCount: 2341,
        consumerLag: 1,
        lastMessage: new Date(Date.now() - 2000).toISOString()
      },
      {
        name: "ss2.observation-track.radar",
        status: "healthy",
        messageCount: 1876,
        consumerLag: 0,
        lastMessage: new Date(Date.now() - 1000).toISOString()
      },
      {
        name: "ss2.association-message.correlation",
        status: "error",
        messageCount: 45,
        consumerLag: 234,
        lastMessage: new Date(Date.now() - 120000).toISOString()
      }
    ]
  },
  {
    name: "Hostility Monitoring (SS5)",
    subsystem: "SS5",
    description: "Threat assessment and hostility analysis", 
    topics: [
      {
        name: "ss5.launch.intent-assessment",
        status: "healthy",
        messageCount: 567,
        consumerLag: 3,
        lastMessage: new Date(Date.now() - 8000).toISOString()
      },
      {
        name: "ss5.pez-wez.predictions.eo",
        status: "healthy",
        messageCount: 234,
        consumerLag: 1,
        lastMessage: new Date(Date.now() - 12000).toISOString()
      },
      {
        name: "ss5.pez-wez.predictions.rf",
        status: "healthy",
        messageCount: 189,
        consumerLag: 0,
        lastMessage: new Date(Date.now() - 7000).toISOString()
      },
      {
        name: "ss5.reentry.assessment",
        status: "warning",
        messageCount: 78,
        consumerLag: 28,
        lastMessage: new Date(Date.now() - 67000).toISOString()
      }
    ]
  },
  {
    name: "Internal Topics",
    subsystem: "INTERNAL",
    description: "AstroShield internal message processing",
    topics: [
      {
        name: "astroshield.event-correlation.events",
        status: "healthy",
        messageCount: 3456,
        consumerLag: 2,
        lastMessage: new Date(Date.now() - 1500).toISOString()
      },
      {
        name: "astroshield.proximity.events",
        status: "healthy",
        messageCount: 987,
        consumerLag: 0,
        lastMessage: new Date(Date.now() - 4000).toISOString()
      },
      {
        name: "astroshield.decision-support.recommendations",
        status: "offline",
        messageCount: 12,
        consumerLag: 456,
        lastMessage: new Date(Date.now() - 300000).toISOString()
      }
    ]
  }
]

export default function TopicHealthDashboard() {
  const [topicCategories, setTopicCategories] = useState<SDATopicCategory[]>(mockTopicCategories)
  const [loading, setLoading] = useState(true)
  const [isUsingSampleData, setIsUsingSampleData] = useState(true)

  const fetchTopicHealth = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/kafka-monitor/topics/health')
      if (response.ok) {
        const data = await response.json()
        if (data.categories && data.categories.length > 0) {
          setTopicCategories(data.categories)
          setIsUsingSampleData(false)
        } else {
          // Use mock data as fallback when API returns empty
          setTopicCategories(mockTopicCategories)
          setIsUsingSampleData(true)
        }
      }
    } catch (error) {
      console.error('Failed to fetch topic health:', error)
      // On error, use mock data as fallback
      setTopicCategories(mockTopicCategories)
      setIsUsingSampleData(true)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTopicHealth()
    
    // Add fallback data if API takes too long or returns empty
    setTimeout(() => {
      if (topicCategories.length === 0 || (isUsingSampleData && !loading)) {
        setTopicCategories(mockTopicCategories)
        setIsUsingSampleData(true)
      }
    }, 3000)
    
    const interval = setInterval(fetchTopicHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600'
      case 'warning': return 'text-yellow-600'
      case 'error': return 'text-red-600'
      case 'offline': return 'text-gray-400'
      default: return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle2 className="h-4 w-4 text-green-600" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-600" />
      case 'error': return <XCircle className="h-4 w-4 text-red-600" />
      case 'offline': return <Clock className="h-4 w-4 text-gray-400" />
      default: return <Activity className="h-4 w-4 text-gray-600" />
    }
  }

  const getSubsystemIcon = (subsystem: string) => {
    switch (subsystem) {
      case 'SS0': return <Satellite className="h-5 w-5 text-blue-600" />
      case 'SS1': return <Target className="h-5 w-5 text-purple-600" />
      case 'SS2': return <Navigation className="h-5 w-5 text-green-600" />
      case 'SS3': return <Command className="h-5 w-5 text-orange-600" />
      case 'SS4': return <Monitor className="h-5 w-5 text-cyan-600" />
      case 'SS5': return <Shield className="h-5 w-5 text-red-600" />
      case 'SS6': return <AlertOctagon className="h-5 w-5 text-pink-600" />
      default: return <Activity className="h-5 w-5 text-gray-600" />
    }
  }

  const getTotalStats = () => {
    const allTopics = topicCategories.flatMap(cat => cat.topics)
    return {
      total: allTopics.length,
      healthy: allTopics.filter(t => t.status === 'healthy').length,
      warning: allTopics.filter(t => t.status === 'warning').length,
      error: allTopics.filter(t => t.status === 'error').length,
      offline: allTopics.filter(t => t.status === 'offline').length,
    }
  }

  const stats = getTotalStats()

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-bold">Topic Health Dashboard</h2>
            {isUsingSampleData && (
              <Badge variant="outline" className="text-xs">
                Sample Data
              </Badge>
            )}
          </div>
          <p className="text-muted-foreground">
            Monitor the health status of all SDA and internal Kafka topics
          </p>
        </div>
        <Button
          onClick={fetchTopicHealth}
          disabled={loading}
          variant="outline"
          size="sm"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm font-medium">Total Topics</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <div>
                <p className="text-sm font-medium">Healthy</p>
                <p className="text-2xl font-bold text-green-600">{stats.healthy}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <div>
                <p className="text-sm font-medium">Warning</p>
                <p className="text-2xl font-bold text-yellow-600">{stats.warning}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <XCircle className="h-4 w-4 text-red-600" />
              <div>
                <p className="text-sm font-medium">Error</p>
                <p className="text-2xl font-bold text-red-600">{stats.error}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-gray-600" />
              <div>
                <p className="text-sm font-medium">Offline</p>
                <p className="text-2xl font-bold text-gray-600">{stats.offline}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {topicCategories.map((category) => (
          <Card key={category.name} className="border-2">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getSubsystemIcon(category.subsystem)}
                  <div>
                    <CardTitle className="text-lg">{category.name}</CardTitle>
                    <p className="text-sm text-muted-foreground">{category.description}</p>
                  </div>
                </div>
                <Badge variant="secondary">
                  {category.topics.length} topics
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                {category.topics.map((topic) => (
                  <div
                    key={topic.name}
                    className="flex items-center justify-between p-3 bg-[#1A1F2E] rounded-lg border border-gray-800"
                  >
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(topic.status)}
                      <div>
                        <p className="text-sm font-medium">{topic.name}</p>
                        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                          <span>{topic.messageCount} msgs</span>
                          <span>Lag: {topic.consumerLag}</span>
                        </div>
                      </div>
                    </div>
                    <Badge variant="outline" className={getStatusColor(topic.status)}>
                      {topic.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {loading && topicCategories.length === 0 && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-5 w-5 animate-spin mr-2" />
          <span>Loading topic health data...</span>
        </div>
      )}
    </div>
  )
} 