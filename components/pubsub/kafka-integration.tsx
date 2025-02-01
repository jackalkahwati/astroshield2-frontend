"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

// This would be replaced with an actual WebSocket connection to your backend
const mockWebSocket = {
  onmessage: null as ((event: { data: string }) => void) | null,
  send: (message: string) => {
    console.log("Sending message:", message)
    // Simulate receiving a message after a short delay
    setTimeout(() => {
      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage({ data: JSON.stringify({ type: "UPDATE", payload: "New satellite data received" }) })
      }
    }, 1000)
  },
}

export function KafkaIntegration() {
  const [messages, setMessages] = useState<string[]>([])

  useEffect(() => {
    mockWebSocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === "UPDATE") {
        setMessages((prev) => [...prev, data.payload])
      }
    }

    // Simulate sending a message to request data
    mockWebSocket.send(JSON.stringify({ type: "SUBSCRIBE", topic: "satellite_data" }))

    return () => {
      mockWebSocket.onmessage = null
    }
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Kafka Integration (PubSub)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {messages.map((message, index) => (
            <div key={index} className="flex items-center space-x-2">
              <Badge variant="secondary">New Data</Badge>
              <span>{message}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

