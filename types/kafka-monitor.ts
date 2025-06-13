export interface KafkaMessage {
  id: string
  topic: string
  direction: 'incoming' | 'outgoing'
  timestamp: string
  messageType: string
  subsystem: string
  sourceSystem: string
  content: any
  size: number
  status: 'success' | 'error' | 'pending'
  latency?: number
  retryCount?: number
  correlationId?: string
}

export interface TopicHealth {
  name: string
  status: 'healthy' | 'warning' | 'error' | 'offline'
  messageCount: number
  lastMessage: string | null
  consumerLag: number
  partitions: number
  replicationFactor: number
  category: 'sda' | 'internal' | 'test'
  subsystem?: string
}

export interface KafkaSubscription {
  id: string
  topic: string
  consumerGroup: string
  status: 'active' | 'inactive' | 'error'
  messageHandler: string
  lastHeartbeat: string
  lag: number
  messagesProcessed: number
  errors: number
}

export interface PerformanceMetric {
  name: string
  value: number
  unit: string
  trend: number
  trendDirection: 'up' | 'down' | 'stable'
  description: string
}

export interface MessageFlowNode {
  id: string
  label: string
  type: 'producer' | 'consumer' | 'topic' | 'subsystem'
  status: 'active' | 'inactive' | 'error'
  messageCount: number
}

export interface MessageFlowEdge {
  from: string
  to: string
  label: string
  messageCount: number
  status: 'active' | 'inactive' | 'error'
}

export interface KafkaMetrics {
  throughput: {
    messagesPerSecond: number
    bytesPerSecond: number
  }
  latency: {
    average: number
    p95: number
    p99: number
  }
  errors: {
    rate: number
    total: number
    recentErrors: Array<{
      timestamp: string
      topic: string
      error: string
    }>
  }
  consumers: {
    totalLag: number
    activeConsumers: number
  }
}

export interface SDATopicCategory {
  name: string
  subsystem: string
  topics: TopicHealth[]
  description: string
}

export interface WebSocketMessage {
  type: 'message' | 'metrics' | 'status' | 'error'
  data: any
  timestamp: string
} 