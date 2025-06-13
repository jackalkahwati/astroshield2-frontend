"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Workflow, Database, Cpu, Activity } from "lucide-react"
import { MessageFlowNode } from "@/types/kafka-monitor"

export default function MessageFlowVisualization() {
  const [nodes, setNodes] = useState<MessageFlowNode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchFlowData = async () => {
      try {
        const response = await fetch('/api/v1/kafka-monitor/message-flow')
        if (response.ok) {
          const data = await response.json()
          setNodes(data.nodes)
        }
      } catch (error) {
        console.error('Failed to fetch flow data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchFlowData()
  }, [])

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'producer': return <Cpu className="h-5 w-5 text-blue-600" />
      case 'consumer': return <Database className="h-5 w-5 text-green-600" />
      case 'topic': return <Workflow className="h-5 w-5 text-purple-600" />
      default: return <Activity className="h-5 w-5 text-gray-600" />
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Message Flow Visualization</h2>
        <p className="text-muted-foreground">
          Visual representation of Kafka message routing
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12">Loading flow diagram...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {nodes.map((node) => (
            <Card key={node.id}>
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  {getNodeIcon(node.type)}
                  <div>
                    <p className="font-medium">{node.label}</p>
                    <p className="text-sm text-muted-foreground">
                      {node.messageCount} messages
                    </p>
                  </div>
                  <Badge variant="outline">{node.status}</Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {!loading && nodes.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Workflow className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Flow Data Available</h3>
            <p className="text-muted-foreground">
              Message flow visualization will appear when Kafka connections are active.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 