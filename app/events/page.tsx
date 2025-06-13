"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, AlertTriangle, Rocket, Orbit, Radio, Compass, ArrowDownCircle, SatelliteIcon } from "lucide-react"
import { formatDate } from "@/lib/utils"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

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
            threat_level: "low"
          }
        ]
        
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

  const highThreatEvents = events.filter(e => e.threat_level === 'high' || e.threat_level === 'severe')
  const processingEvents = events.filter(e => e.status === 'processing' || e.status === 'detected')

  const metrics = [
    { title: "Total Events", value: dashboardData.total_events.toString() },
    { title: "Processing", value: processingEvents.length.toString() },
    { title: "High Threats", value: highThreatEvents.length.toString() },
    { title: "Completed", value: (dashboardData.events_by_status.completed || 0).toString() },
  ]

  if (!isLoaded) {
    return (
      <div className="space-y-6">
        <p>Loading events...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Events</h1>
          <p className="text-muted-foreground">Space event detection and analysis</p>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card key={metric.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{metric.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metric.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content */}
      <Tabs defaultValue="recent" className="space-y-4">
        <TabsList>
          <TabsTrigger value="recent">Recent Events</TabsTrigger>
          <TabsTrigger value="threats">High Priority</TabsTrigger>
          <TabsTrigger value="details">Event Details</TabsTrigger>
        </TabsList>

        <TabsContent value="recent" className="space-y-4">
          <div className="grid gap-4">
            {events.map((event) => (
              <Card key={event.id} className="cursor-pointer hover:bg-gray-50" onClick={() => setSelectedEvent(event)}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {EVENT_ICONS[event.event_type]}
                      <span className="capitalize">{event.event_type.replace('_', ' ')}</span>
                    </div>
                    <div className="flex gap-2">
                      <Badge className={STATUS_COLORS[event.status]}>
                        {event.status}
                      </Badge>
                      {event.threat_level && (
                        <Badge className={THREAT_COLORS[event.threat_level]}>
                          {event.threat_level}
                        </Badge>
                      )}
                    </div>
                  </CardTitle>
                  <CardDescription>Object: {event.object_id} • ID: {event.id}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p><strong>Created:</strong> {formatDate(event.creation_time)}</p>
                      <p><strong>Updated:</strong> {formatDate(event.update_time)}</p>
                    </div>
                    {event.coa_recommendation && (
                      <div>
                        <p><strong>Priority:</strong> {event.coa_recommendation.priority}/5</p>
                        <p className="text-muted-foreground">{event.coa_recommendation.description}</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="threats" className="space-y-4">
          <div className="grid gap-4">
            {highThreatEvents.length > 0 ? (
              highThreatEvents.map((event) => (
                <Card key={event.id} className="border-orange-200">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                        <span className="capitalize">{event.event_type.replace('_', ' ')}</span>
                      </div>
                      <Badge className={THREAT_COLORS[event.threat_level || 'none']}>
                        {event.threat_level}
                      </Badge>
                    </CardTitle>
                    <CardDescription>Object: {event.object_id}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {event.coa_recommendation && (
                      <div className="space-y-2">
                        <p className="font-medium">{event.coa_recommendation.title}</p>
                        <p className="text-sm text-muted-foreground">{event.coa_recommendation.description}</p>
                        <div className="space-y-1">
                          <p className="text-sm font-medium">Recommended Actions:</p>
                          <ul className="text-sm text-muted-foreground list-disc list-inside">
                            {event.coa_recommendation.actions.map((action, index) => (
                              <li key={index}>{action}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card>
                <CardContent className="text-center py-8">
                  <p className="text-muted-foreground">No high-threat events currently</p>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="details" className="space-y-4">
          {selectedEvent ? (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {EVENT_ICONS[selectedEvent.event_type]}
                  Event Details: {selectedEvent.id}
                </CardTitle>
                <CardDescription>
                  {selectedEvent.event_type.charAt(0).toUpperCase() + selectedEvent.event_type.slice(1)} event for object {selectedEvent.object_id}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Event Information</h4>
                    <div className="space-y-1 text-sm">
                      <p><strong>Type:</strong> {selectedEvent.event_type}</p>
                      <p><strong>Status:</strong> {selectedEvent.status}</p>
                      <p><strong>Threat Level:</strong> {selectedEvent.threat_level || 'None'}</p>
                      <p><strong>Object ID:</strong> {selectedEvent.object_id}</p>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">Timeline</h4>
                    <div className="space-y-1 text-sm">
                      <p><strong>Created:</strong> {formatDate(selectedEvent.creation_time)}</p>
                      <p><strong>Last Updated:</strong> {formatDate(selectedEvent.update_time)}</p>
                    </div>
                  </div>
                </div>
                
                {selectedEvent.coa_recommendation && (
                  <div>
                    <h4 className="font-medium mb-2">Course of Action</h4>
                    <div className="space-y-2">
                      <p className="font-medium">{selectedEvent.coa_recommendation.title}</p>
                      <p className="text-sm text-muted-foreground">{selectedEvent.coa_recommendation.description}</p>
                      <p className="text-sm"><strong>Priority:</strong> {selectedEvent.coa_recommendation.priority}/5</p>
                      <div>
                        <p className="text-sm font-medium mb-1">Recommended Actions:</p>
                        <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                          {selectedEvent.coa_recommendation.actions.map((action, index) => (
                            <li key={index}>{action}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">Select an event to view details</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}