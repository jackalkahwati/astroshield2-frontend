'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Activity,
  Pause,
  Play,
  Filter,
  Download,
  AlertTriangle,
  Zap,
  Clock,
  Server
} from 'lucide-react'
import { AstroShieldWebSocket, WebSocketMessage } from '@/lib/websocket-client'
import { themeColors } from '@/lib/theme-colors'

interface KafkaMessage {
  topic: string
  partition: number
  offset: number
  key: string | null
  value: any
  timestamp: string
  headers?: Record<string, string>
  metadata?: {
    producerId?: string
    compressionType?: string
    serializedKeySize?: number
    serializedValueSize?: number
  }
}

interface RealTimeKafkaStreamProps {
  topics?: string[]
  maxMessages?: number
  autoScroll?: boolean
  className?: string
}

export default function RealTimeKafkaStream({
  topics = [],
  maxMessages = 100,
  autoScroll = true,
  className = ""
}: RealTimeKafkaStreamProps) {
  const [messages, setMessages] = useState<KafkaMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [selectedTopics, setSelectedTopics] = useState<string[]>(topics.length > 0 ? topics : [
    'ss5.launch.detection',
    'ss5.threat.correlation',
    'ss5.pez-wez.kkv'
  ])
  const [stats, setStats] = useState({
    messagesPerSecond: 0,
    totalMessages: 0,
    lastMessageTime: null as Date | null
  })
  
  const wsRef = useRef<AstroShieldWebSocket | null>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messageBufferRef = useRef<KafkaMessage[]>([])
  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Available SS5 Kafka topics
  const availableTopics = [
    'ss5.launch.detection',
    'ss5.launch.intent',
    'ss5.launch.assessment',
    'ss5.pez-wez.eo',
    'ss5.pez-wez.rf',
    'ss5.pez-wez.kkv',
    'ss5.pez-wez.grappler',
    'ss5.pez-wez.conjunction',
    'ss5.reentry.prediction',
    'ss5.reentry.assessment',
    'ss5.asat.capability',
    'ss5.threat.correlation'
  ]

  // Filter messages based on selected topics
  const filteredMessages = messages.filter(message => 
    selectedTopics.length === 0 || selectedTopics.includes(message.topic)
  )

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new AstroShieldWebSocket({
      url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/kafka',
      reconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    })

    wsRef.current = ws

    ws.on('connected', () => {
      setIsConnected(true)
      // Subscribe to selected topics
      if (selectedTopics.length > 0) {
        ws.subscribe(selectedTopics)
      }
    })

    ws.on('disconnected', () => {
      setIsConnected(false)
    })

    ws.on('kafka_message', (message: KafkaMessage) => {
      if (!isPaused) {
        messageBufferRef.current.push(message)
        
        // Update stats
        setStats(prev => ({
          messagesPerSecond: prev.messagesPerSecond,
          totalMessages: prev.totalMessages + 1,
          lastMessageTime: new Date()
        }))
      }
    })

    ws.on('error', (error) => {
      console.error('WebSocket error:', error)
    })

    // Connect to WebSocket
    ws.connect().catch(error => {
      console.error('Failed to connect to WebSocket:', error)
      // Fall back to sample data if connection fails
      startSampleDataStream()
    })

    // Update messages from buffer periodically
    const updateInterval = setInterval(() => {
      if (messageBufferRef.current.length > 0) {
        setMessages(prev => {
          const newMessages = [...prev, ...messageBufferRef.current]
          messageBufferRef.current = []
          
          // Keep only the most recent messages
          if (newMessages.length > maxMessages) {
            return newMessages.slice(-maxMessages)
          }
          return newMessages
        })

        // Auto-scroll to bottom
        if (autoScroll && scrollAreaRef.current) {
          setTimeout(() => {
            const scrollContainer = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]')
            if (scrollContainer) {
              scrollContainer.scrollTop = scrollContainer.scrollHeight
            }
          }, 100)
        }
      }
    }, 100)

    // Calculate messages per second
    statsIntervalRef.current = setInterval(() => {
      const currentTime = Date.now()
      const recentMessages = filteredMessages.filter(msg => {
        const msgTime = new Date(msg.timestamp).getTime()
        return currentTime - msgTime <= 1000
      })
      
      setStats(prev => ({
        ...prev,
        messagesPerSecond: recentMessages.length
      }))
    }, 1000)

    return () => {
      clearInterval(updateInterval)
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
      }
      if (wsRef.current) {
        wsRef.current.disconnect()
      }
    }
  }, [])

  // Handle topic selection changes
  useEffect(() => {
    if (wsRef.current?.isConnected() && selectedTopics.length > 0) {
      wsRef.current.unsubscribe(availableTopics)
      wsRef.current.subscribe(selectedTopics)
    }
  }, [selectedTopics])

  // Sample data stream fallback
  const startSampleDataStream = () => {
    const sampleMessages = [
      {
        topic: 'ss5.launch.detection',
        value: {
          launch_id: 'LNC-2024-001',
          detection_time: new Date().toISOString(),
          launch_site: { lat: 28.5721, lon: -80.6480 },
          vehicle_type: 'Unknown',
          confidence: 0.92
        }
      },
      {
        topic: 'ss5.pez-wez.kkv',
        value: {
          threat_id: 'KKV-2024-042',
          target_id: 'USA-123',
          pez_entry_time: new Date(Date.now() + 300000).toISOString(),
          wez_entry_time: new Date(Date.now() + 600000).toISOString(),
          intercept_probability: 0.78
        }
      },
      {
        topic: 'ss5.threat.correlation',
        value: {
          correlation_id: 'CORR-2024-089',
          related_events: ['LNC-2024-001', 'KKV-2024-042'],
          threat_level: 'HIGH',
          recommended_actions: ['ACTIVATE_DEFENSES', 'NOTIFY_COMMAND']
        }
      },
      {
        topic: 'ss5.launch.assessment',
        value: {
          launch_id: 'LNC-2024-001',
          assessment_time: new Date().toISOString(),
          vehicle_classification: 'BALLISTIC_MISSILE',
          trajectory_assessment: 'THREAT_TRAJECTORY',
          impact_probability: 0.85
        }
      },
      {
        topic: 'ss5.pez-wez.rf',
        value: {
          threat_id: 'RF-2024-078',
          target_id: 'USA-456',
          frequency_band: '2.4GHz',
          signal_strength: -65,
          jamming_probability: 0.73
        }
      },
      {
        topic: 'ss5.reentry.prediction',
        value: {
          object_id: 'DEBRIS-2024-012',
          predicted_reentry: new Date(Date.now() + 86400000).toISOString(),
          impact_zone: { lat: 35.2271, lon: -80.8431 },
          uncertainty_ellipse: '50km x 25km'
        }
      }
    ]

    let messageIndex = 0
    const interval = setInterval(() => {
      if (!isPaused && !isConnected) {
        const template = sampleMessages[messageIndex % sampleMessages.length]
        const message: KafkaMessage = {
          topic: template.topic,
          partition: Math.floor(Math.random() * 3),
          offset: Math.floor(Math.random() * 10000),
          key: null,
          value: template.value,
          timestamp: new Date().toISOString(),
          headers: {
            'message-type': 'sample',
            'source': 'demo-producer'
          },
          metadata: {
            producerId: 'astroshield-demo',
            compressionType: 'none',
            serializedKeySize: 0,
            serializedValueSize: JSON.stringify(template.value).length
          }
        }

        messageBufferRef.current.push(message)
        messageIndex++
      }
    }, 800 + Math.random() * 1500)

    return () => clearInterval(interval)
  }

  const handleTopicToggle = (topic: string) => {
    setSelectedTopics(prev =>
      prev.includes(topic)
        ? prev.filter(t => t !== topic)
        : [...prev, topic]
    )
  }

  const handleSelectAll = () => {
    setSelectedTopics(availableTopics)
  }

  const handleClearAll = () => {
    setSelectedTopics([])
  }

  const handleExport = () => {
    const data = JSON.stringify(filteredMessages, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `kafka-messages-${new Date().toISOString()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const getTopicColor = (topic: string) => {
    if (topic.includes('launch')) return 'bg-red-500'
    if (topic.includes('pez-wez')) return 'bg-orange-500'
    if (topic.includes('reentry')) return 'bg-yellow-500'
    if (topic.includes('threat')) return 'bg-purple-500'
    if (topic.includes('asat')) return 'bg-pink-500'
    return 'bg-blue-500'
  }

  return (
    <Card className={`bg-[#1A1F2E] border-[#374151] ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-white">
            <Activity className="h-5 w-5" />
            Real-Time Kafka Stream
            {isConnected ? (
              <Badge variant="default" className="ml-2 bg-green-600 text-white">
                <Zap className="h-3 w-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="secondary" className="ml-2 bg-[#2A2F3E] text-[#D1D5DB]">
                <Server className="h-3 w-3 mr-1" />
                Demo Mode
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="bg-[#2A2F3E] border-[#4B5563] text-white">
              {stats.messagesPerSecond} msg/s
            </Badge>
            <Badge variant="outline" className="bg-[#2A2F3E] border-[#4B5563] text-white">
              {filteredMessages.length} total
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPaused(!isPaused)}
              className="bg-[#2A2F3E] border-[#4B5563] text-white hover:bg-[#3A3F4E]"
            >
              {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExport}
              className="bg-[#2A2F3E] border-[#4B5563] text-white hover:bg-[#3A3F4E]"
            >
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Topic Filter */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-white" />
              <span className="text-sm font-medium text-white">Filter Topics:</span>
              <Badge variant="outline" className="bg-[#2A2F3E] border-[#4B5563] text-[#D1D5DB]">
                {selectedTopics.length} of {availableTopics.length} selected
              </Badge>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSelectAll}
                className="bg-[#2A2F3E] border-[#4B5563] text-white hover:bg-[#3A3F4E] text-xs"
              >
                Select All
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearAll}
                className="bg-[#2A2F3E] border-[#4B5563] text-white hover:bg-[#3A3F4E] text-xs"
              >
                Clear All
              </Button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {availableTopics.map(topic => (
              <Badge
                key={topic}
                variant={selectedTopics.includes(topic) ? "default" : "outline"}
                className={`cursor-pointer transition-all ${
                  selectedTopics.includes(topic)
                    ? 'bg-[#1E40AF] text-white border-[#3B82F6] hover:bg-[#1D4ED8]'
                    : 'bg-[#2A2F3E] border-[#4B5563] text-[#D1D5DB] hover:bg-[#3A3F4E] hover:border-[#3B82F6]'
                }`}
                onClick={() => handleTopicToggle(topic)}
              >
                {topic.replace('ss5.', '')}
              </Badge>
            ))}
          </div>
        </div>

        {/* Message Stream */}
        <ScrollArea className="h-96 w-full rounded-lg border bg-[#0A0E1A] border-[#374151] p-4" ref={scrollAreaRef}>
          {filteredMessages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-[#9CA3AF]">
              <AlertTriangle className="h-8 w-8 mb-2" />
              <p className="text-white">No messages to display</p>
              <p className="text-sm">
                {selectedTopics.length === 0 
                  ? 'Select topics to filter messages'
                  : 'Waiting for messages from selected topics...'
                }
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredMessages.map((message, index) => (
                <div
                  key={`${message.topic}-${message.offset}-${index}`}
                  className="bg-[#1A1F2E] rounded-lg p-3 shadow-sm border-2 border-[#374151] hover:border-[#4B5563] transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${getTopicColor(message.topic)}`} />
                      <span className="text-sm font-medium text-white">{message.topic}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-[#D1D5DB]">
                      <Clock className="h-3 w-3" />
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  
                  <div className="text-xs space-y-1">
                    <div className="flex gap-4 text-[#D1D5DB]">
                      <span>Partition: {message.partition}</span>
                      <span>Offset: {message.offset}</span>
                      {message.metadata?.serializedValueSize && (
                        <span>Size: {message.metadata.serializedValueSize}B</span>
                      )}
                    </div>
                    
                    <div className="bg-[#0A0E1A] rounded p-2 mt-2 border border-[#374151]">
                      <pre className="text-xs overflow-x-auto text-[#E5E7EB]">
                        {JSON.stringify(message.value, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        {/* Connection Status */}
        {!isConnected && (
          <div className="mt-4 p-3 bg-[#1A1F2E] border-2 border-yellow-600 rounded-lg">
            <div className="flex items-center gap-2 text-yellow-400">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm">
                Running in demo mode. Connect to a Kafka broker for live data.
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 