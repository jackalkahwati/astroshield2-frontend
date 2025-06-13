'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea
} from 'recharts'
import {
  Clock,
  ZoomIn,
  ZoomOut,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Activity,
  AlertTriangle,
  Info
} from 'lucide-react'

interface TimelineEvent {
  id: string
  timestamp: Date
  type: string
  objectId: string
  confidence: number
  uncertainty: number
  expertValidator?: string
  correlatedEvents?: string[]
  observables: string[]
}

interface TimeRange {
  start: Date
  end: Date
}

export default function OrbitalTimelineVisualization() {
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [timeRange, setTimeRange] = useState<TimeRange>({
    start: new Date(Date.now() - 24 * 60 * 60 * 1000),
    end: new Date()
  })
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [zoomLevel, setZoomLevel] = useState(1)

  // Generate sample timeline events
  useEffect(() => {
    const generateEvents = () => {
      const eventTypes = [
        { type: 'maneuver_start', validator: 'Tom Johnson', uncertainty: 2.1 },
        { type: 'proximity_raan_drift', validator: 'Nathan Parrott', uncertainty: 4.5 },
        { type: 'signature_rcs_change', validator: 'Nathan Parrott', uncertainty: 5.8 },
        { type: 'breakup_fragmentation', validator: 'Moriba Jah', uncertainty: 8.1 },
        { type: 'discovery_post_deployment', validator: 'Jim Shell', uncertainty: 6.3 }
      ]

      const newEvents: TimelineEvent[] = []
      const numEvents = 50

      for (let i = 0; i < numEvents; i++) {
        const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)]
        const timestamp = new Date(
          timeRange.start.getTime() + 
          Math.random() * (timeRange.end.getTime() - timeRange.start.getTime())
        )

        newEvents.push({
          id: `EVT-${i.toString().padStart(3, '0')}`,
          timestamp,
          type: eventType.type,
          objectId: `OBJ-${Math.floor(Math.random() * 99999)}`,
          confidence: 70 + Math.random() * 30,
          uncertainty: eventType.uncertainty,
          expertValidator: eventType.validator,
          correlatedEvents: Math.random() > 0.7 ? 
            [`EVT-${Math.floor(Math.random() * numEvents).toString().padStart(3, '0')}`] : 
            undefined,
          observables: ['trajectory', 'brightness', 'rf_signature'].slice(0, Math.floor(Math.random() * 3) + 1)
        })
      }

      setEvents(newEvents.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()))
    }

    generateEvents()
  }, [timeRange])

  // Playback animation
  useEffect(() => {
    if (!isPlaying) return

    const interval = setInterval(() => {
      setCurrentTime(prev => {
        const newTime = new Date(prev.getTime() + 60000 * playbackSpeed)
        if (newTime > timeRange.end) {
          setIsPlaying(false)
          return timeRange.end
        }
        return newTime
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isPlaying, playbackSpeed, timeRange.end])

  // Prepare data for timeline chart
  const timelineData = events.map(event => ({
    time: event.timestamp.getTime(),
    displayTime: event.timestamp.toLocaleTimeString(),
    [event.type]: event.confidence,
    uncertainty: event.uncertainty,
    id: event.id
  }))

  // Group events by hour for density visualization
  const eventDensityData = () => {
    const hourlyBuckets: { [key: string]: number } = {}
    
    events.forEach(event => {
      const hourKey = new Date(
        event.timestamp.getFullYear(),
        event.timestamp.getMonth(),
        event.timestamp.getDate(),
        event.timestamp.getHours()
      ).toISOString()
      
      hourlyBuckets[hourKey] = (hourlyBuckets[hourKey] || 0) + 1
    })

    return Object.entries(hourlyBuckets).map(([time, count]) => ({
      time: new Date(time).getTime(),
      displayTime: new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      density: count
    })).sort((a, b) => a.time - b.time)
  }

  const handleZoom = (direction: 'in' | 'out') => {
    const center = new Date(
      (timeRange.start.getTime() + timeRange.end.getTime()) / 2
    )
    const currentSpan = timeRange.end.getTime() - timeRange.start.getTime()
    const newSpan = direction === 'in' ? currentSpan / 2 : currentSpan * 2
    
    setTimeRange({
      start: new Date(center.getTime() - newSpan / 2),
      end: new Date(center.getTime() + newSpan / 2)
    })
    setZoomLevel(prev => direction === 'in' ? prev * 2 : prev / 2)
  }

  const getEventColor = (type: string) => {
    const colors: { [key: string]: string } = {
      'maneuver_start': '#ef4444',
      'proximity_raan_drift': '#f97316',
      'signature_rcs_change': '#3b82f6',
      'breakup_fragmentation': '#8b5cf6',
      'discovery_post_deployment': '#10b981'
    }
    return colors[type] || '#6b7280'
  }

  const densityData = eventDensityData()

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Orbital Event Timeline</h2>
          <p className="text-sm text-gray-600 mt-1">
            Interactive timeline with Moriba Jah uncertainty quantification
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleZoom('out')}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Badge variant="outline">
            {zoomLevel}x
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleZoom('in')}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Playback Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Timeline Playback
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentTime(timeRange.start)}
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                variant={isPlaying ? "default" : "outline"}
                size="sm"
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentTime(timeRange.end)}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
              
              <div className="flex items-center gap-2 ml-4">
                <span className="text-sm">Speed:</span>
                <Slider
                  value={[playbackSpeed]}
                  onValueChange={([value]) => setPlaybackSpeed(value)}
                  min={0.5}
                  max={10}
                  step={0.5}
                  className="w-32"
                />
                <span className="text-sm font-medium">{playbackSpeed}x</span>
              </div>
              
              <div className="ml-auto text-sm font-medium">
                {currentTime.toLocaleString()}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Event Density Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Event Density Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={densityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="displayTime"
                  interval="preserveStartEnd"
                />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="density" 
                  stroke="#3b82f6" 
                  fill="#3b82f6" 
                  fillOpacity={0.6}
                  name="Events per Hour"
                />
                <ReferenceLine 
                  x={currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  stroke="#ef4444"
                  strokeWidth={2}
                  label="Current Time"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Interactive Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Event Timeline with Uncertainty Bands</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Event Type Legend */}
            <div className="flex flex-wrap gap-2">
              {['maneuver_start', 'proximity_raan_drift', 'signature_rcs_change', 'breakup_fragmentation', 'discovery_post_deployment'].map(type => (
                <Badge 
                  key={type}
                  style={{ backgroundColor: getEventColor(type) }}
                  className="text-white"
                >
                  {type.replace(/_/g, ' ')}
                </Badge>
              ))}
            </div>

            {/* Timeline Visualization */}
            <div className="relative h-96 overflow-x-auto">
              <div className="absolute inset-0 bg-gray-50 rounded-lg p-4">
                {events.map((event, index) => {
                  const xPosition = ((event.timestamp.getTime() - timeRange.start.getTime()) / 
                    (timeRange.end.getTime() - timeRange.start.getTime())) * 100
                  const yPosition = (index % 10) * 35 + 20

                  return (
                    <div
                      key={event.id}
                      className="absolute cursor-pointer group"
                      style={{
                        left: `${xPosition}%`,
                        top: `${yPosition}px`
                      }}
                      onClick={() => setSelectedEvent(event)}
                    >
                      {/* Uncertainty band */}
                      <div
                        className="absolute opacity-20"
                        style={{
                          backgroundColor: getEventColor(event.type),
                          width: `${event.uncertainty * 10}px`,
                          height: '20px',
                          left: `-${event.uncertainty * 5}px`,
                          top: '-2px'
                        }}
                      />
                      
                      {/* Event marker */}
                      <div
                        className="w-4 h-4 rounded-full border-2 border-white shadow-lg"
                        style={{ backgroundColor: getEventColor(event.type) }}
                      />
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                        <div className="bg-black text-white text-xs rounded py-1 px-2 whitespace-nowrap">
                          <div className="font-medium">{event.type}</div>
                          <div>{event.objectId}</div>
                          <div>±{event.uncertainty}% uncertainty</div>
                        </div>
                      </div>
                    </div>
                  )
                })}

                {/* Current time indicator */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-red-500"
                  style={{
                    left: `${((currentTime.getTime() - timeRange.start.getTime()) / 
                      (timeRange.end.getTime() - timeRange.start.getTime())) * 100}%`
                  }}
                />
              </div>
            </div>

            {/* Selected Event Details */}
            {selectedEvent && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Event Details: {selectedEvent.id}</AlertTitle>
                <AlertDescription>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    <div>
                      <span className="font-medium">Type:</span> {selectedEvent.type}
                    </div>
                    <div>
                      <span className="font-medium">Object:</span> {selectedEvent.objectId}
                    </div>
                    <div>
                      <span className="font-medium">Confidence:</span> {selectedEvent.confidence.toFixed(1)}%
                    </div>
                    <div>
                      <span className="font-medium">Uncertainty:</span> ±{selectedEvent.uncertainty}%
                    </div>
                    <div>
                      <span className="font-medium">Time:</span> {selectedEvent.timestamp.toLocaleString()}
                    </div>
                    <div>
                      <span className="font-medium">Validator:</span> {selectedEvent.expertValidator}
                    </div>
                  </div>
                  {selectedEvent.correlatedEvents && (
                    <div className="mt-2">
                      <span className="font-medium">Correlated Events:</span> {selectedEvent.correlatedEvents.join(', ')}
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 