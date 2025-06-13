"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  ArrowDown, 
  ArrowUp, 
  Search, 
  Filter, 
  Pause, 
  Play, 
  Eye,
  Clock,
  MessageSquare,
  Server,
  AlertCircle,
  CheckCircle2
} from "lucide-react"
import { KafkaMessage, WebSocketMessage } from "@/types/kafka-monitor"

// Mock data for demo/fallback
const mockMessages: KafkaMessage[] = [
  {
    id: "msg-001",
    topic: "ss5.launch.intent-assessment",
    messageType: "LaunchIntentAssessment", 
    direction: "incoming",
    status: "success",
    timestamp: new Date(Date.now() - 5000).toISOString(),
    size: 2341,
    subsystem: "SS5",
    sourceSystem: "HUNTER-1",
    correlationId: "corr-launch-001",
    latency: 45,
    content: {
      header: {
        timestamp: new Date().toISOString(),
        message_id: "LAUNCH-20250106-001",
        source_system: "HUNTER-1"
      },
      launch_event: {
        vehicle_id: "FALCON-9-001",
        launch_site: "CAPE-CANAVERAL",
        threat_level: "MODERATE"
      }
    }
  },
  {
    id: "msg-002", 
    topic: "ss5.pez-wez.predictions",
    messageType: "PEZWEZPrediction",
    direction: "outgoing",
    status: "success", 
    timestamp: new Date(Date.now() - 12000).toISOString(),
    size: 1876,
    subsystem: "SS5",
    sourceSystem: "ASTROSHIELD-1",
    correlationId: "corr-pez-002",
    latency: 23,
    content: {
      header: {
        timestamp: new Date().toISOString(),
        message_id: "PEZ-20250106-002"
      },
      predictions: {
        weapon_type: "KKV",
        engagement_zone: "LEO-ZONE-A",
        confidence: 0.87
      }
    }
  },
  {
    id: "msg-003",
    topic: "ss2.state-vector.full",
    messageType: "StateVector",
    direction: "incoming", 
    status: "pending",
    timestamp: new Date(Date.now() - 18000).toISOString(),
    size: 3124,
    subsystem: "SS2",
    sourceSystem: "SPACE-FENCE",
    correlationId: "corr-state-003",
    latency: 67,
    content: {
      header: {
        timestamp: new Date().toISOString(),
        message_id: "STATE-20250106-003"
      },
      state_vector: {
        satellite_id: "SAT-12345",
        position: [6678000, 0, 0],
        velocity: [0, 7546, 0]
      }
    }
  },
  {
    id: "msg-004",
    topic: "ss0.launch-detection.radar",
    messageType: "LaunchDetection",
    direction: "incoming",
    status: "error",
    timestamp: new Date(Date.now() - 25000).toISOString(), 
    size: 987,
    subsystem: "SS0",
    sourceSystem: "RADAR-SITE-2",
    latency: 156,
    content: {
      error: "Message validation failed",
      original_message: "corrupted_data"
    }
  },
  {
    id: "msg-005",
    topic: "ss5.reentry.assessment", 
    messageType: "ReentryAssessment",
    direction: "outgoing",
    status: "success",
    timestamp: new Date(Date.now() - 30000).toISOString(),
    size: 2156,
    subsystem: "SS5",
    sourceSystem: "ASTROSHIELD-1", 
    correlationId: "corr-reentry-005",
    latency: 34,
    content: {
      header: {
        timestamp: new Date().toISOString(),
        message_id: "REENTRY-20250106-005"
      },
      assessment: {
        object_id: "DEB-67890",
        reentry_time: "2025-01-07T14:30:00Z",
        impact_zone: "PACIFIC-OCEAN"
      }
    }
  }
]

export default function LiveMessageStream() {
  const [messages, setMessages] = useState<KafkaMessage[]>(mockMessages)
  const [filteredMessages, setFilteredMessages] = useState<KafkaMessage[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedDirection, setSelectedDirection] = useState<'all' | 'incoming' | 'outgoing'>('all')
  const [selectedMessage, setSelectedMessage] = useState<KafkaMessage | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isUsingSampleData, setIsUsingSampleData] = useState(true)
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // WebSocket connection for real-time messages
  useEffect(() => {
    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/api/v1/kafka-monitor/live-stream`
      
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }
      
      wsRef.current.onmessage = (event) => {
        try {
          const wsMessage: WebSocketMessage = JSON.parse(event.data)
          
          if (wsMessage.type === 'message' && !isPaused) {
            const newMessage: KafkaMessage = wsMessage.data
            // Switch to real data when messages start flowing
            setIsUsingSampleData(false)
            setMessages(prev => [newMessage, ...prev.slice(0, 499)]) // Keep last 500 messages
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      wsRef.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected')
        // Fall back to sample data when disconnected
        if (messages.length === 0 || isUsingSampleData) {
          setMessages(mockMessages)
          setIsUsingSampleData(true)
        }
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000)
      }
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }
    }

    connectWebSocket()

    // Fallback to sample data if no real messages arrive within 5 seconds
    const fallbackTimeout = setTimeout(() => {
      if (messages.length <= mockMessages.length && isUsingSampleData) {
        console.log('No real messages received, using sample data')
        setMessages(mockMessages)
        setIsUsingSampleData(true)
      }
    }, 5000)

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      clearTimeout(fallbackTimeout)
    }
  }, [isPaused])

  // Filter messages based on search term and direction
  useEffect(() => {
    let filtered = messages

    if (selectedDirection !== 'all') {
      filtered = filtered.filter(msg => msg.direction === selectedDirection)
    }

    if (searchTerm) {
      filtered = filtered.filter(msg => 
        msg.topic.toLowerCase().includes(searchTerm.toLowerCase()) ||
        msg.messageType.toLowerCase().includes(searchTerm.toLowerCase()) ||
        msg.subsystem.toLowerCase().includes(searchTerm.toLowerCase()) ||
        msg.sourceSystem.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    setFilteredMessages(filtered)
  }, [messages, searchTerm, selectedDirection])

  // Auto-scroll to bottom unless user has scrolled up
  useEffect(() => {
    if (!isPaused && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [filteredMessages, isPaused])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'bg-green-100 text-green-800'
      case 'error': return 'bg-red-100 text-red-800'
      case 'pending': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getDirectionIcon = (direction: string) => {
    return direction === 'incoming' ? 
      <ArrowDown className="h-4 w-4 text-blue-600" /> : 
      <ArrowUp className="h-4 w-4 text-green-600" />
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle2 className="h-4 w-4 text-green-600" />
      case 'error': return <AlertCircle className="h-4 w-4 text-red-600" />
      default: return <Clock className="h-4 w-4 text-yellow-600" />
    }
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
  }

  const incomingMessages = filteredMessages.filter(m => m.direction === 'incoming')
  const outgoingMessages = filteredMessages.filter(m => m.direction === 'outgoing')

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-5 w-5" />
              <CardTitle>Live Message Stream</CardTitle>
              <Badge variant={isConnected ? "default" : "destructive"}>
                {isConnected ? "Connected" : "Disconnected"}
              </Badge>
              {isUsingSampleData && (
                <Badge variant="outline" className="text-xs">
                  Sample Data
                </Badge>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                variant={isPaused ? "default" : "outline"}
                onClick={() => setIsPaused(!isPaused)}
              >
                {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                {isPaused ? "Resume" : "Pause"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search messages..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <select
                value={selectedDirection}
                onChange={(e) => setSelectedDirection(e.target.value as any)}
                className="border border-input bg-background px-3 py-2 text-sm rounded-md"
              >
                <option value="all">All Messages</option>
                <option value="incoming">Incoming Only</option>
                <option value="outgoing">Outgoing Only</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Message Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <ArrowDown className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm font-medium">Incoming</p>
                <p className="text-2xl font-bold">{incomingMessages.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <ArrowUp className="h-4 w-4 text-green-600" />
              <div>
                <p className="text-sm font-medium">Outgoing</p>
                <p className="text-2xl font-bold">{outgoingMessages.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-4 w-4 text-purple-600" />
              <div>
                <p className="text-sm font-medium">Total</p>
                <p className="text-2xl font-bold">{filteredMessages.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Server className="h-4 w-4 text-orange-600" />
              <div>
                <p className="text-sm font-medium">Active Topics</p>
                <p className="text-2xl font-bold">
                  {new Set(filteredMessages.map(m => m.topic)).size}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Messages List */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Messages</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px]">
            <div className="space-y-2">
              {filteredMessages.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  {isPaused ? "Stream paused" : isUsingSampleData ? "Showing sample data - connect to Kafka for live messages" : "No messages to display"}
                </div>
              ) : (
                filteredMessages.map((message) => (
                  <div
                    key={message.id}
                    className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                    onClick={() => setSelectedMessage(message)}
                  >
                    <div className="flex items-center space-x-2">
                      {getDirectionIcon(message.direction)}
                      {getStatusIcon(message.status)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <p className="text-sm font-medium truncate">{message.topic}</p>
                        <Badge variant="outline" className="text-xs">
                          {message.subsystem}
                        </Badge>
                        <Badge className={getStatusColor(message.status)}>
                          {message.status}
                        </Badge>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-muted-foreground mt-1">
                        <span>{message.messageType}</span>
                        <span>{message.sourceSystem}</span>
                        <span>{formatSize(message.size)}</span>
                        {message.latency && <span>{message.latency}ms</span>}
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground text-right">
                      <p>{formatTime(message.timestamp)}</p>
                      <Button variant="ghost" size="sm">
                        <Eye className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Message Detail Modal */}
      {selectedMessage && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-4xl max-h-[80vh] overflow-hidden">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Message Details</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedMessage(null)}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[60vh]">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold">Message Info</h4>
                      <div className="space-y-1 text-sm">
                        <p><strong>Topic:</strong> {selectedMessage.topic}</p>
                        <p><strong>Type:</strong> {selectedMessage.messageType}</p>
                        <p><strong>Direction:</strong> {selectedMessage.direction}</p>
                        <p><strong>Status:</strong> {selectedMessage.status}</p>
                        <p><strong>Size:</strong> {formatSize(selectedMessage.size)}</p>
                        <p><strong>Timestamp:</strong> {selectedMessage.timestamp}</p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold">Source Info</h4>
                      <div className="space-y-1 text-sm">
                        <p><strong>Subsystem:</strong> {selectedMessage.subsystem}</p>
                        <p><strong>Source System:</strong> {selectedMessage.sourceSystem}</p>
                        {selectedMessage.correlationId && (
                          <p><strong>Correlation ID:</strong> {selectedMessage.correlationId}</p>
                        )}
                        {selectedMessage.latency && (
                          <p><strong>Latency:</strong> {selectedMessage.latency}ms</p>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2">Message Content</h4>
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto">
                      {JSON.stringify(selectedMessage.content, null, 2)}
                    </pre>
                  </div>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
} 