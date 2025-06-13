"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { 
  Play, 
  Pause, 
  Square,
  Search,
  Settings,
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Users,
  RefreshCw
} from "lucide-react"
import { KafkaSubscription } from "@/types/kafka-monitor"

export default function SubscriptionManager() {
  const [subscriptions, setSubscriptions] = useState<KafkaSubscription[]>([])
  const [filteredSubscriptions, setFilteredSubscriptions] = useState<KafkaSubscription[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedStatus, setSelectedStatus] = useState<'all' | 'active' | 'inactive' | 'error'>('all')

  const fetchSubscriptions = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/kafka-monitor/subscriptions')
      if (response.ok) {
        const data = await response.json()
        setSubscriptions(data.subscriptions)
      }
    } catch (error) {
      console.error('Failed to fetch subscriptions:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSubscriptions()
    const interval = setInterval(fetchSubscriptions, 15000) // Update every 15 seconds
    return () => clearInterval(interval)
  }, [])

  // Filter subscriptions
  useEffect(() => {
    let filtered = subscriptions

    if (selectedStatus !== 'all') {
      filtered = filtered.filter(sub => sub.status === selectedStatus)
    }

    if (searchTerm) {
      filtered = filtered.filter(sub => 
        sub.topic.toLowerCase().includes(searchTerm.toLowerCase()) ||
        sub.consumerGroup.toLowerCase().includes(searchTerm.toLowerCase()) ||
        sub.messageHandler.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    setFilteredSubscriptions(filtered)
  }, [subscriptions, searchTerm, selectedStatus])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800'
      case 'inactive': return 'bg-gray-100 text-gray-800'
      case 'error': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle2 className="h-4 w-4 text-green-600" />
      case 'inactive': return <Clock className="h-4 w-4 text-gray-600" />
      case 'error': return <AlertCircle className="h-4 w-4 text-red-600" />
      default: return <Activity className="h-4 w-4 text-gray-600" />
    }
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const handleSubscriptionAction = async (subscriptionId: string, action: 'start' | 'stop' | 'restart') => {
    try {
      const response = await fetch(`/api/v1/kafka-monitor/subscriptions/${subscriptionId}/${action}`, {
        method: 'POST'
      })
      if (response.ok) {
        fetchSubscriptions() // Refresh the list
      }
    } catch (error) {
      console.error(`Failed to ${action} subscription:`, error)
    }
  }

  const getStats = () => {
    return {
      total: subscriptions.length,
      active: subscriptions.filter(s => s.status === 'active').length,
      inactive: subscriptions.filter(s => s.status === 'inactive').length,
      error: subscriptions.filter(s => s.status === 'error').length,
      totalLag: subscriptions.reduce((sum, s) => sum + s.lag, 0),
      totalProcessed: subscriptions.reduce((sum, s) => sum + s.messagesProcessed, 0)
    }
  }

  const stats = getStats()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Subscription Manager</h2>
          <p className="text-muted-foreground">
            Monitor and manage active Kafka subscriptions and consumer groups
          </p>
        </div>
        <Button
          onClick={fetchSubscriptions}
          disabled={loading}
          variant="outline"
          size="sm"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Settings className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm font-medium">Total</p>
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
                <p className="text-sm font-medium">Active</p>
                <p className="text-2xl font-bold text-green-600">{stats.active}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-gray-600" />
              <div>
                <p className="text-sm font-medium">Inactive</p>
                <p className="text-2xl font-bold text-gray-600">{stats.inactive}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <div>
                <p className="text-sm font-medium">Errors</p>
                <p className="text-2xl font-bold text-red-600">{stats.error}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Users className="h-4 w-4 text-purple-600" />
              <div>
                <p className="text-sm font-medium">Total Lag</p>
                <p className="text-2xl font-bold">{stats.totalLag.toLocaleString()}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-orange-600" />
              <div>
                <p className="text-sm font-medium">Processed</p>
                <p className="text-2xl font-bold">{stats.totalProcessed.toLocaleString()}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search subscriptions..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Status:</span>
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value as any)}
                className="border border-input bg-background px-3 py-2 text-sm rounded-md"
              >
                <option value="all">All</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="error">Error</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Subscriptions List */}
      <Card>
        <CardHeader>
          <CardTitle>Active Subscriptions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {filteredSubscriptions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                {loading ? "Loading subscriptions..." : "No subscriptions found"}
              </div>
            ) : (
              filteredSubscriptions.map((subscription) => (
                <div
                  key={subscription.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(subscription.status)}
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <p className="text-sm font-medium truncate">{subscription.topic}</p>
                        <Badge className={getStatusColor(subscription.status)}>
                          {subscription.status}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center space-x-6 text-xs text-muted-foreground mt-1">
                        <span><strong>Group:</strong> {subscription.consumerGroup}</span>
                        <span><strong>Handler:</strong> {subscription.messageHandler}</span>
                        <span><strong>Lag:</strong> {subscription.lag}</span>
                        <span><strong>Processed:</strong> {subscription.messagesProcessed.toLocaleString()}</span>
                        {subscription.errors > 0 && (
                          <span className="text-red-600"><strong>Errors:</strong> {subscription.errors}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <div className="text-right text-xs text-muted-foreground">
                      <p>Last heartbeat:</p>
                      <p>{formatTime(subscription.lastHeartbeat)}</p>
                    </div>
                    
                    <div className="flex items-center space-x-1">
                      {subscription.status === 'active' ? (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleSubscriptionAction(subscription.id, 'stop')}
                        >
                          <Pause className="h-3 w-3" />
                        </Button>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleSubscriptionAction(subscription.id, 'start')}
                        >
                          <Play className="h-3 w-3" />
                        </Button>
                      )}
                      
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleSubscriptionAction(subscription.id, 'restart')}
                      >
                        <RefreshCw className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Loading State */}
      {loading && subscriptions.length === 0 && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-5 w-5 animate-spin mr-2" />
          <span>Loading subscriptions...</span>
        </div>
      )}
    </div>
  )
} 