import { create } from 'zustand'
import type { StoreApi } from 'zustand'
import { useEffect } from 'react'

type MessageType = 'ccdm_update' | 'thermal_signature' | 'shape_change' | 'propulsive_event' | 'error' | 'ping' | 'pong'

// Add specific payload types
interface CCDMUpdatePayload {
  object_id: string
  assessment_type: string
  results: Record<string, any>
  confidence_level: number
  recommendations: string[]
}

interface ThermalSignaturePayload {
  object_id: string
  temperature_kelvin: number
  heat_signature_pattern: string
  emission_spectrum: Record<string, number>
  anomaly_score: number
}

interface ShapeChangePayload {
  object_id: string
  volume_change: number
  surface_area_change: number
  aspect_ratio_change: number
  confidence: number
}

interface PropulsiveEventPayload {
  object_id: string
  event_type: string
  delta_v: number
  direction: [number, number, number]
  confidence: number
}

// Update WebSocketMessage to use specific payload types
interface WebSocketMessage<T = any> {
  type: MessageType
  payload: T extends 'ccdm_update' ? CCDMUpdatePayload :
          T extends 'thermal_signature' ? ThermalSignaturePayload :
          T extends 'shape_change' ? ShapeChangePayload :
          T extends 'propulsive_event' ? PropulsiveEventPayload :
          T extends 'error' ? { message: string } :
          T extends 'ping' | 'pong' ? { timestamp: number } :
          never
  timestamp: string
}

type MessageHandler = (message: WebSocketMessage) => void

// Add message buffering and queue
interface MessageQueue {
  message: Omit<WebSocketMessage, 'timestamp'>
  attempts: number
  lastAttempt: number
}

// Add dead letter queue and priority types
interface DeadLetterMessage extends MessageQueue {
  failureReason: string
  lastError: string
  timestamp: string
}

interface PrioritizedMessage extends MessageQueue {
  priority: 'high' | 'normal' | 'low'
  category: 'telemetry' | 'command' | 'status'
}

interface WebSocketState {
  connected: boolean
  messages: WebSocketMessage[]
  connect: () => void
  disconnect: () => void
  send: (message: Omit<WebSocketMessage, 'timestamp'>) => void
  error: string | null
  clearError: () => void
  subscribe: (handler: MessageHandler) => () => void
  subscribeToCCDM: (handler: (payload: CCDMUpdatePayload) => void) => () => void
  subscribeToThermal: (handler: (payload: ThermalSignaturePayload) => void) => () => void
  subscribeToShapeChange: (handler: (payload: ShapeChangePayload) => void) => () => void
  subscribeToPropulsive: (handler: (payload: PropulsiveEventPayload) => void) => () => void
  messageQueue: MessageQueue[]
  retryMessage: (message: MessageQueue) => void
  clearQueue: () => void
  deadLetterQueue: DeadLetterMessage[]
  getDeadLetterMessages: () => DeadLetterMessage[]
  retryDeadLetter: (messageId: string) => void
  clearDeadLetterQueue: () => void
}

const MAX_RECONNECT_ATTEMPTS = 5
const INITIAL_RECONNECT_DELAY = 1000
const MAX_RECONNECT_DELAY = 30000
const HEARTBEAT_INTERVAL = 30000

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001'

// Add message validation types
type ValidationResult = {
  valid: boolean
  errors?: string[]
}

// Add validation functions for each message type
const validators = {
  ccdm_update: (payload: unknown): payload is CCDMUpdatePayload => {
    const p = payload as CCDMUpdatePayload
    return (
      typeof p.object_id === 'string' &&
      typeof p.assessment_type === 'string' &&
      typeof p.confidence_level === 'number' &&
      Array.isArray(p.recommendations)
    )
  },
  
  thermal_signature: (payload: unknown): payload is ThermalSignaturePayload => {
    const p = payload as ThermalSignaturePayload
    return (
      typeof p.object_id === 'string' &&
      typeof p.temperature_kelvin === 'number' &&
      typeof p.heat_signature_pattern === 'string' &&
      typeof p.anomaly_score === 'number'
    )
  },
  
  shape_change: (payload: unknown): payload is ShapeChangePayload => {
    const p = payload as ShapeChangePayload
    return (
      typeof p.object_id === 'string' &&
      typeof p.volume_change === 'number' &&
      typeof p.surface_area_change === 'number' &&
      typeof p.aspect_ratio_change === 'number' &&
      typeof p.confidence === 'number'
    )
  },
  
  propulsive_event: (payload: unknown): payload is PropulsiveEventPayload => {
    const p = payload as PropulsiveEventPayload
    return (
      typeof p.object_id === 'string' &&
      typeof p.event_type === 'string' &&
      typeof p.delta_v === 'number' &&
      Array.isArray(p.direction) &&
      p.direction.length === 3 &&
      typeof p.confidence === 'number'
    )
  }
}

export const useWebSocketStore = create<WebSocketState>((
  set: StoreApi<WebSocketState>['setState'],
  get: StoreApi<WebSocketState>['getState']
) => {
  let ws: WebSocket | null = null
  let reconnectAttempts = 0
  let reconnectTimeout: NodeJS.Timeout | null = null
  let heartbeatInterval: NodeJS.Timeout | null = null
  let messageHandlers: Set<MessageHandler> = new Set()
  let messageQueue: MessageQueue[] = []
  const MAX_QUEUE_SIZE = 100
  const MAX_MESSAGE_ATTEMPTS = 3
  const MESSAGE_RETRY_DELAY = 5000
  let deadLetterQueue: DeadLetterMessage[] = []

  const clearHeartbeat = () => {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval)
      heartbeatInterval = null
    }
  }

  const startHeartbeat = () => {
    clearHeartbeat()
    heartbeatInterval = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ 
          type: 'ping' as const,
          payload: { timestamp: Date.now() }
        }))
      }
    }, HEARTBEAT_INTERVAL)
  }

  const addToDeadLetterQueue = (
    message: MessageQueue,
    failureReason: string,
    error: string
  ) => {
    const deadLetter: DeadLetterMessage = {
      ...message,
      failureReason,
      lastError: error,
      timestamp: new Date().toISOString()
    }
    deadLetterQueue.push(deadLetter)
    set({ deadLetterQueue })
  }

  const processQueue = () => {
    const now = Date.now()
    messageQueue = messageQueue.filter(item => {
      if (item.attempts >= MAX_MESSAGE_ATTEMPTS) {
        addToDeadLetterQueue(
          item,
          'Max retry attempts exceeded',
          'Message failed after maximum retries'
        )
        return false
      }
      
      // Prioritize messages based on type
      const priority = getPriority(item.message)
      const retryDelay = getRetryDelay(priority, item.attempts)
      
      if (now - item.lastAttempt >= retryDelay) {
        if (ws?.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify(item.message))
            return false // Remove from queue if sent successfully
          } catch (error) {
            item.attempts++
            item.lastAttempt = now
            if (error instanceof Error) {
              console.error(`Failed to send message: ${error.message}`)
            }
          }
        }
      }
      return true
    })
    
    set({ messageQueue })
  }

  const getPriority = (message: Omit<WebSocketMessage, 'timestamp'>): 'high' | 'normal' | 'low' => {
    switch (message.type) {
      case 'propulsive_event':
      case 'ccdm_update':
        return 'high'
      case 'thermal_signature':
      case 'shape_change':
        return 'normal'
      default:
        return 'low'
    }
  }

  const getRetryDelay = (priority: 'high' | 'normal' | 'low', attempts: number): number => {
    const baseDelay = MESSAGE_RETRY_DELAY
    const maxDelay = 30000 // 30 seconds

    switch (priority) {
      case 'high':
        return Math.min(baseDelay * Math.pow(1.5, attempts), maxDelay)
      case 'normal':
        return Math.min(baseDelay * Math.pow(2, attempts), maxDelay)
      case 'low':
        return Math.min(baseDelay * Math.pow(3, attempts), maxDelay)
    }
  }

  const clearDeadLetterQueue = () => {
    deadLetterQueue = []
    set({ deadLetterQueue })
  }

  const retryDeadLetter = (messageId: string) => {
    const message = deadLetterQueue.find(m => 
      m.message.type + m.timestamp === messageId
    )
    if (message) {
      queueMessage(message.message)
      deadLetterQueue = deadLetterQueue.filter(m => 
        m.message.type + m.timestamp !== messageId
      )
      set({ deadLetterQueue })
    }
  }

  const queueProcessor = setInterval(processQueue, 1000)

  const connect = () => {
    try {
      if (ws?.readyState === WebSocket.OPEN) return

      ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        console.log('WebSocket connected')
        set({ connected: true, error: null })
        reconnectAttempts = 0
        startHeartbeat()
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        set({ connected: false })
        clearHeartbeat()
        
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(
            INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts),
            MAX_RECONNECT_DELAY
          )
          
          reconnectTimeout = setTimeout(() => {
            reconnectAttempts++
            connect()
          }, delay)
        } else {
          set({ error: 'Maximum reconnection attempts reached' })
        }
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage
          
          if (message.type === 'pong') return
          
          // Validate message payload
          if (message.type in validators) {
            const validator = validators[message.type as keyof typeof validators]
            if (!validator(message.payload)) {
              console.error('Invalid message payload:', message)
              set({ error: `Invalid ${message.type} payload` })
              return
            }
          }
          
          const timestampedMessage = {
            ...message,
            timestamp: new Date().toISOString()
          }

          set((state) => ({
            messages: [...state.messages, timestampedMessage]
          }))

          // Notify handlers with validated payload
          messageHandlers.forEach(handler => handler(timestampedMessage))

          if (message.type === 'error') {
            set({ error: message.payload.message })
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
          set({ error: 'Failed to parse message' })
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        set({ error: 'WebSocket connection error' })
      }
    } catch (error) {
      console.error('Error connecting to WebSocket:', error)
      set({ error: 'Failed to establish WebSocket connection' })
    }
  }

  const disconnect = () => {
    if (ws) {
      ws.close()
      ws = null
    }
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    if (queueProcessor) {
      clearInterval(queueProcessor)
    }
    clearHeartbeat()
    reconnectAttempts = 0
    messageHandlers.clear()
    clearQueue()
    set({ connected: false, error: null })
  }

  const send = (message: Omit<WebSocketMessage, 'timestamp'>) => {
    if (ws?.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message))
      } catch (error) {
        queueMessage(message)
      }
    } else {
      queueMessage(message)
    }
  }

  const queueMessage = (message: Omit<WebSocketMessage, 'timestamp'>) => {
    if (messageQueue.length >= MAX_QUEUE_SIZE) {
      console.warn('Message queue full, dropping oldest message')
      messageQueue.shift()
    }
    
    messageQueue.push({
      message,
      attempts: 0,
      lastAttempt: Date.now()
    })
    
    set({ messageQueue })
  }

  const clearQueue = () => {
    messageQueue = []
    set({ messageQueue })
  }

  const subscribe = (handler: MessageHandler) => {
    messageHandlers.add(handler)
    return () => {
      messageHandlers.delete(handler)
    }
  }

  const clearError = () => set({ error: null })

  const subscribeToCCDM = (handler: (payload: CCDMUpdatePayload) => void) => {
    const messageHandler = (message: WebSocketMessage) => {
      if (message.type === 'ccdm_update') {
        handler(message.payload as CCDMUpdatePayload)
      }
    }
    messageHandlers.add(messageHandler)
    return () => messageHandlers.delete(messageHandler)
  }

  const subscribeToThermal = (handler: (payload: ThermalSignaturePayload) => void) => {
    const messageHandler = (message: WebSocketMessage) => {
      if (message.type === 'thermal_signature') {
        handler(message.payload as ThermalSignaturePayload)
      }
    }
    messageHandlers.add(messageHandler)
    return () => messageHandlers.delete(messageHandler)
  }

  const subscribeToShapeChange = (handler: (payload: ShapeChangePayload) => void) => {
    const messageHandler = (message: WebSocketMessage) => {
      if (message.type === 'shape_change') {
        handler(message.payload as ShapeChangePayload)
      }
    }
    messageHandlers.add(messageHandler)
    return () => messageHandlers.delete(messageHandler)
  }

  const subscribeToPropulsive = (handler: (payload: PropulsiveEventPayload) => void) => {
    const messageHandler = (message: WebSocketMessage) => {
      if (message.type === 'propulsive_event') {
        handler(message.payload as PropulsiveEventPayload)
      }
    }
    messageHandlers.add(messageHandler)
    return () => messageHandlers.delete(messageHandler)
  }

  return {
    connected: false,
    messages: [],
    error: null,
    connect,
    disconnect,
    send,
    clearError,
    subscribe,
    subscribeToCCDM,
    subscribeToThermal,
    subscribeToShapeChange,
    subscribeToPropulsive,
    messageQueue,
    clearQueue,
    retryMessage: (message: MessageQueue) => {
      message.attempts = 0
      message.lastAttempt = 0
      messageQueue.push(message)
      set({ messageQueue })
    },
    deadLetterQueue,
    getDeadLetterMessages: () => deadLetterQueue,
    retryDeadLetter,
    clearDeadLetterQueue
  }
})

// Hook for components to use WebSocket
export function useWebSocket() {
  const store = useWebSocketStore()
  
  useEffect(() => {
    store.connect()
    return () => store.disconnect()
  }, [])

  return {
    connected: store.connected,
    error: store.error,
    clearError: store.clearError,
    sendMessage: store.send,
    subscribe: store.subscribe,
    subscribeToCCDM: store.subscribeToCCDM,
    subscribeToThermal: store.subscribeToThermal,
    subscribeToShapeChange: store.subscribeToShapeChange,
    subscribeToPropulsive: store.subscribeToPropulsive,
    messageQueue: store.messageQueue,
    clearQueue: store.clearQueue,
    retryMessage: store.retryMessage,
    deadLetterQueue: store.deadLetterQueue,
    retryDeadLetter: store.retryDeadLetter,
    clearDeadLetterQueue: store.clearDeadLetterQueue
  }
} 