'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Activity, ArrowRight, AlertTriangle } from 'lucide-react'

interface EventFlowDiagramProps {
  events?: any[]
  className?: string
}

const EventFlowDiagram: React.FC<EventFlowDiagramProps> = ({ 
  events = [], 
  className = "" 
}) => {
  // Sample event flow data
  const sampleEvents = [
    {
      id: "SS5-20241221-001",
      type: "Launch Detection",
      timestamp: "2024-12-21T10:30:00Z",
      status: "active",
      confidence: 0.94
    },
    {
      id: "SS5-20241221-002", 
      type: "Intent Assessment",
      timestamp: "2024-12-21T10:31:15Z",
      status: "processing",
      confidence: 0.87
    },
    {
      id: "SS5-20241221-003",
      type: "Threat Classification",
      timestamp: "2024-12-21T10:32:30Z", 
      status: "pending",
      confidence: 0.72
    }
  ]

  const displayEvents = events.length > 0 ? events : sampleEvents

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Event Flow Diagram
          <Badge variant="outline" className="ml-auto">
            {displayEvents.length} Events
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {displayEvents.map((event, index) => (
            <div key={event.id} className="flex items-center gap-3">
              <div className="flex-shrink-0">
                <div className={`w-3 h-3 rounded-full ${
                  event.status === 'active' ? 'bg-green-500' :
                  event.status === 'processing' ? 'bg-yellow-500' : 
                  'bg-gray-400'
                }`} />
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {event.type}
                  </p>
                  <Badge variant={
                    event.status === 'active' ? 'default' :
                    event.status === 'processing' ? 'secondary' :
                    'outline'
                  }>
                    {event.confidence ? `${(event.confidence * 100).toFixed(1)}%` : event.status}
                  </Badge>
                </div>
                <p className="text-sm text-gray-500">
                  {event.id} • {new Date(event.timestamp).toLocaleTimeString()}
                </p>
              </div>

              {index < displayEvents.length - 1 && (
                <ArrowRight className="h-4 w-4 text-gray-400 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>

        {displayEvents.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>No event flow data available</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default EventFlowDiagram 