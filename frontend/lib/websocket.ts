import { create, StateCreator } from 'zustand'
import { useEffect } from 'react'

interface WebSocketStore {
  socket: WebSocket | null
  connected: boolean
  reconnectAttempts: number
  maxReconnectAttempts: number
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => void
}

const WEBSOCKET_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001'
const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_DELAY = 1000 // 1 second

export const useWebSocketStore = create<WebSocketStore>((set: any, get: any) => ({
  socket: null,
  connected: false,
  reconnectAttempts: 0,
  maxReconnectAttempts: MAX_RECONNECT_ATTEMPTS,

  connect: () => {
    const { socket, reconnectAttempts, maxReconnectAttempts } = get()

    if (socket?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WEBSOCKET_URL)

    ws.onopen = () => {
      console.log('WebSocket connected')
      set({ socket: ws, connected: true, reconnectAttempts: 0 })
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      set({ socket: null, connected: false })

      // Attempt to reconnect
      if (reconnectAttempts < maxReconnectAttempts) {
        setTimeout(() => {
          set({ reconnectAttempts: reconnectAttempts + 1 })
          get().connect()
        }, RECONNECT_DELAY * Math.pow(2, reconnectAttempts))
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleWebSocketMessage(data)
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    set({ socket: ws })
  },

  disconnect: () => {
    const { socket } = get()
    if (socket) {
      socket.close()
      set({ socket: null, connected: false })
    }
  },

  sendMessage: (message: any) => {
    const { socket, connected } = get()
    if (socket && connected) {
      socket.send(JSON.stringify(message))
    }
  },
}))

function handleWebSocketMessage(data: any) {
  switch (data.type) {
    case 'ccdm_update':
      handleCCDMUpdate(data.payload)
      break
    case 'thermal_signature':
      handleThermalSignature(data.payload)
      break
    case 'shape_change':
      handleShapeChange(data.payload)
      break
    case 'propulsive_event':
      handlePropulsiveEvent(data.payload)
      break
    default:
      console.warn('Unknown message type:', data.type)
  }
}

function handleCCDMUpdate(payload: any) {
  // Update CCDM charts and data
  const event = new CustomEvent('ccdm-update', { detail: payload })
  window.dispatchEvent(event)
}

function handleThermalSignature(payload: any) {
  // Update thermal signature visualizations
  const event = new CustomEvent('thermal-update', { detail: payload })
  window.dispatchEvent(event)
}

function handleShapeChange(payload: any) {
  // Update shape change visualizations
  const event = new CustomEvent('shape-update', { detail: payload })
  window.dispatchEvent(event)
}

function handlePropulsiveEvent(payload: any) {
  // Update propulsive event visualizations
  const event = new CustomEvent('propulsion-update', { detail: payload })
  window.dispatchEvent(event)
}

// Hook for components to use WebSocket
export function useWebSocket() {
  const store = useWebSocketStore()
  
  useEffect(() => {
    store.connect()
    return () => store.disconnect()
  }, [])

  return {
    connected: store.connected,
    sendMessage: store.sendMessage,
  }
} 