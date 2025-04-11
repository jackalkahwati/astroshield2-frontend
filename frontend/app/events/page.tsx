"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, AlertTriangle, Rocket, Orbit, Radio, Compass, ArrowDownCircle, SatelliteIcon } from "lucide-react"
import { formatDate } from "@/lib/utils"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"

// Event types mapping to icons
const EVENT_ICONS: Record<string, React.ReactNode> = {
  launch: <Rocket className="h-5 w-5" />,
  reentry: <ArrowDownCircle className="h-5 w-5" />,
  maneuver: <Compass className="h-5 w-5" />,
  separation: <SatelliteIcon className="h-5 w-5" />,
  proximity: <Orbit className="h-5 w-5" />,
  link_change: <Radio className="h-5 w-5" />,
  attitude_change: <Activity className="h-5 w-5" />
}

// Status color mapping
const STATUS_COLORS: Record<string, string> = {
  detected: "bg-blue-100 text-blue-800",
  processing: "bg-orange-100 text-orange-800",
  awaiting_data: "bg-purple-100 text-purple-800",
  completed: "bg-green-100 text-green-800",
  rejected: "bg-gray-100 text-gray-800",
  error: "bg-red-100 text-red-800"
}

// Threat level color mapping
const THREAT_COLORS: Record<string, string> = {
  none: "bg-gray-100 text-gray-800",
  low: "bg-blue-100 text-blue-800",
  moderate: "bg-yellow-100 text-yellow-800",
  high: "bg-orange-100 text-orange-800",
  severe: "bg-red-100 text-red-800"
}

export default function EventsPage() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [dashboardData, setDashboardData] = useState({
    total_events: 0,
    events_by_type: {} as Record<string, number>,
    events_by_status: {} as Record<string, number>,
    events_by_threat: {} as Record<string, number>,
    recent_high_threats: [] as any[]
  })
  
  interface EventData {
    id: string;
    event_type: string;
    object_id: string;
    status: string;
    creation_time: string;
    update_time: string;
    threat_level: string | null;
    coa_recommendation?: {
      title: string;
      description: string;
      priority: number;
      actions: string[];
    };
  }
  
  const [events, setEvents] = useState<EventData[]>([])
  const [selectedEvent, setSelectedEvent] = useState<EventData | null>(null)
  
  // Fetch dashboard data and events
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // In a real implementation, these would be API calls
        // const response = await fetch('/api/v1/events/dashboard')
        // const data = await response.json()
        
        // Mock dashboard data
        const mockDashboardData = {
          total_events: 42,
          events_by_type: {
            launch: 8,
            reentry: 5,
            maneuver: 12,
            separation: 3,
            proximity: 9,
            link_change: 3,
            attitude_change: 2
          },
          events_by_status: {
            detected: 5,
            processing: 3,
            awaiting_data: 2,
            completed: 28,
            rejected: 1,
            error: 3
          },
          events_by_threat: {
            none: 15,
            low: 17,
            moderate: 7,
            high: 3,
            severe: 0
          },
          recent_high_threats: []
        }
        
        // Mock events data
        const mockEvents = [
          {
            id: "evt-12345abc",
            event_type: "maneuver",
            object_id: "sat-78901",
            status: "completed",
            creation_time: "2025-03-22T15:42:23Z",
            update_time: "2025-03-22T15:45:12Z",
            threat_level: "moderate",
            coa_recommendation: {
              title: "Maneuver Response Plan: Moderate Threat",
              description: "Response plan for maneuver by object sat-78901",
              priority: 3,
              actions: [
                "Continue monitoring spacecraft",
                "Increase tracking frequency",
                "Alert affected spacecraft operators",
                "Assess new orbit for strategic implications"
              ]
            }
          },
          {
            id: "evt-67890def",
            event_type: "proximity",
            object_id: "sat-12345",
            status: "completed",
            creation_time: "2025-03-22T14:18:05Z",
            update_time: "2025-03-22T14:22:31Z",
            threat_level: "high",
            coa_recommendation: {
              title: "Proximity Response Plan: High Threat",
              description: "Response plan for close approach event between sat-12345 and sat-67890",
              priority: 5,
              actions: [
                "Continue monitoring both spacecraft",
                "Prepare evasive maneuver options",
                "Alert operators immediately",
                "Establish communication with object owners",
                "Prepare diplomatic channels if necessary"
              ]
            }
          },
          {
            id: "evt-abcde123",
            event_type: "launch",
            object_id: "sat-45678",
            status: "processing",
            creation_time: "2025-03-23T08:15:00Z",
            update_time: "2025-03-23T08:16:22Z",
            threat_level: null
          },
          {
            id: "evt-fghij456",
            event_type: "reentry",
            object_id: "sat-98765",
            status: "completed",
            creation_time: "2025-03-21T22:45:17Z",
            update_time: "2025-03-21T23:01:45Z",
            threat_level: "low",
            coa_recommendation: {
              title: "Reentry Response Plan: Low Threat",
              description: "Response plan for reentry of object sat-98765",
              priority: 2,
              actions: [
                "Track trajectory to impact/endpoint",
                "Notify air traffic control if necessary",
                "Document reentry parameters"
              ]
            }
          },
          {
            id: "evt-klmno789",
            event_type: "link_change",
            object_id: "sat-23456",
            status: "awaiting_data",
            creation_time: "2025-03-23T10:05:33Z",
            update_time: "2025-03-23T10:06:11Z",
            threat_level: null
          }
        ]
        
        // Update the state with mock data
        setDashboardData(mockDashboardData)
        setEvents(mockEvents)
        setSelectedEvent(mockEvents[0])
        setIsLoaded(true)
      } catch (error) {
        console.error("Error fetching events data:", error)
      }
    }
    
    fetchDashboardData()
  }, [])

  if (!isLoaded) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-center">
          <p className="text-xl font-semibold">Loading events...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Welder's Arc Events</h1>
        <Badge variant="outline" className="text-sm">
          Events: {dashboardData.total_events}
        </Badge>
      </div>

      <Tabs defaultValue="dashboard" className="w-full">
        <TabsList>
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="events">Event List</TabsTrigger>
          <TabsTrigger value="detail">Event Detail</TabsTrigger>
        </TabsList>
        
        <TabsContent value="dashboard" className="space-y-4">
          {/* Event Type Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Event Type Distribution</CardTitle>
              <CardDescription>
                Count of events by discrete event type
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(dashboardData.events_by_type).map(([type, count]) => (
                  <div key={type} className="flex flex-col items-center justify-center p-4 border rounded-lg">
                    <div className="mb-2">
                      {EVENT_ICONS[type] || <Activity className="h-5 w-5" />}
                    </div>
                    <div className="text-xl font-bold">{count}</div>
                    <div className="text-sm text-gray-500 capitalize">{type.replace('_', ' ')}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Status and Threat Level */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Status Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Processing Status</CardTitle>
                <CardDescription>
                  Current status of events in the system
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(dashboardData.events_by_status).map(([status, count]) => {
                    const percentage = Math.round((count / dashboardData.total_events) * 100)
                    return (
                      <div key={status} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{status.replace('_', ' ')}</span>
                          <span>{count} ({percentage}%)</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
            
            {/* Threat Level Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Threat Level Assessment</CardTitle>
                <CardDescription>
                  Distribution of event threat levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(dashboardData.events_by_threat).map(([level, count]) => {
                    const percentage = Math.round((count / dashboardData.total_events) * 100)
                    return (
                      <div key={level} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{level}</span>
                          <span>{count} ({percentage}%)</span>
                        </div>
                        <Progress 
                          value={percentage} 
                          className={`h-2 ${level === 'high' || level === 'severe' ? 'bg-red-200' : ''}`}
                        />
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="events" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Event List</CardTitle>
              <CardDescription>
                Recent events detected by the system
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="px-4 py-2 text-left">Type</th>
                      <th className="px-4 py-2 text-left">Object ID</th>
                      <th className="px-4 py-2 text-left">Time</th>
                      <th className="px-4 py-2 text-left">Status</th>
                      <th className="px-4 py-2 text-left">Threat Level</th>
                      <th className="px-4 py-2 text-left">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((event) => (
                      <tr 
                        key={event.id} 
                        className="border-b hover:bg-gray-50 cursor-pointer"
                        onClick={() => setSelectedEvent(event)}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            {EVENT_ICONS[event.event_type] || <Activity className="h-4 w-4" />}
                            <span className="capitalize">{event.event_type.replace('_', ' ')}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">{event.object_id}</td>
                        <td className="px-4 py-3">{formatDate(event.creation_time)}</td>
                        <td className="px-4 py-3">
                          <span className={`capitalize px-2 py-1 rounded-full text-xs ${STATUS_COLORS[event.status]}`}>
                            {event.status.replace('_', ' ')}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {event.threat_level ? (
                            <span className={`capitalize px-2 py-1 rounded-full text-xs ${THREAT_COLORS[event.threat_level]}`}>
                              {event.threat_level}
                            </span>
                          ) : (
                            <span className="text-xs text-gray-500">Pending</span>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              setSelectedEvent(event)
                            }}
                          >
                            View
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="detail" className="space-y-4">
          {selectedEvent ? (
            <Card>
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center gap-2">
                      {EVENT_ICONS[selectedEvent.event_type] || <Activity className="h-5 w-5" />}
                      <CardTitle className="capitalize">
                        {selectedEvent.event_type.replace('_', ' ')} Event
                      </CardTitle>
                    </div>
                    <CardDescription>
                      Event ID: {selectedEvent.id} | Object: {selectedEvent.object_id}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <span className={`capitalize px-2 py-1 rounded-full text-xs ${STATUS_COLORS[selectedEvent.status]}`}>
                      {selectedEvent.status.replace('_', ' ')}
                    </span>
                    {selectedEvent.threat_level && (
                      <span className={`capitalize px-2 py-1 rounded-full text-xs ${THREAT_COLORS[selectedEvent.threat_level]}`}>
                        {selectedEvent.threat_level}
                      </span>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Event Details</h3>
                    <div className="space-y-2">
                      <div className="grid grid-cols-2 gap-2 border-b pb-2">
                        <span className="text-sm font-medium">Detection Time:</span>
                        <span className="text-sm">{formatDate(selectedEvent.creation_time)}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 border-b pb-2">
                        <span className="text-sm font-medium">Last Updated:</span>
                        <span className="text-sm">{formatDate(selectedEvent.update_time)}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 border-b pb-2">
                        <span className="text-sm font-medium">Status:</span>
                        <span className="text-sm capitalize">{selectedEvent.status.replace('_', ' ')}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 border-b pb-2">
                        <span className="text-sm font-medium">Threat Level:</span>
                        <span className="text-sm capitalize">{selectedEvent.threat_level || "Pending"}</span>
                      </div>
                    </div>
                  </div>
                  
                  {selectedEvent.coa_recommendation && (
                    <div>
                      <h3 className="text-lg font-medium mb-2">Course of Action</h3>
                      <div className="space-y-2">
                        <div>
                          <span className="text-sm font-medium">Title:</span>
                          <p className="text-sm">{selectedEvent.coa_recommendation?.title}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium">Description:</span>
                          <p className="text-sm">{selectedEvent.coa_recommendation?.description}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium">Priority:</span>
                          <div className="flex items-center gap-1 mt-1">
                            {[...Array(5)].map((_, i) => (
                              <div 
                                key={i} 
                                className={`w-5 h-2 rounded ${i < (selectedEvent.coa_recommendation?.priority || 0) ? 'bg-red-500' : 'bg-gray-200'}`} 
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                {selectedEvent.coa_recommendation && (
                  <div>
                    <h3 className="text-lg font-medium mb-2">Recommended Actions</h3>
                    <ul className="space-y-1 list-disc list-inside">
                      {selectedEvent.coa_recommendation?.actions?.map((action, index) => (
                        <li key={index} className="text-sm">{action}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="flex justify-end mt-4">
                  <Button 
                    variant="outline" 
                    onClick={() => console.log("Process event manually")}
                    disabled={selectedEvent.status === "completed"}
                  >
                    {selectedEvent.status === "completed" ? "Processing Complete" : "Process Event"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <p className="text-lg text-gray-500">No event selected. Please select an event from the Event List.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}