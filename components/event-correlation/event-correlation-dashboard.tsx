"use client"

import React, { useState, useEffect, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Search, 
  GitBranch, 
  Clock, 
  AlertTriangle, 
  CheckCircle,
  XCircle,
  Activity,
  Layers,
  Network,
  Filter,
  Download,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Zap,
  Link2
} from 'lucide-react'
import { format } from 'date-fns'

// Dynamic imports for heavy visualization components
const EventFlowDiagram = dynamic(() => import('./event-flow-diagram'), { 
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">Loading visualization...</div>
})

const EventTimelineView = ({ event }: { event: any }) => (
        <div className="h-64 flex items-center justify-center border-2 border-dashed border-gray-600 rounded-lg bg-[#1A1F2E]">
    <div className="text-center">
      <Activity className="h-8 w-8 mx-auto mb-2 text-gray-400" />
      <p className="text-gray-600">Timeline View</p>
      <p className="text-sm text-gray-500">Temporal visualization coming soon</p>
    </div>
  </div>
)

// Utility function for safe date formatting
const formatDateSafe = (dateString: string | null | undefined, formatStr: string, fallback: string = 'Invalid time'): string => {
  try {
    if (!dateString) return fallback
    const date = new Date(dateString)
    if (isNaN(date.getTime())) return fallback
    return format(date, formatStr)
  } catch (error) {
    console.error('Date formatting error:', error)
    return fallback
  }
}

interface CorrelatedEvent {
  event_id: string
  event_type: string
  object_id: string
  creation_time: string
  status: string
  message_count: number
  subsystem_flow: string[]
  metadata: Record<string, any>
  parents: CorrelatedEvent[]
  children: CorrelatedEvent[]
}

interface EventBottleneck {
  event_id: string
  object_id: string
  event_type: string
  creation_time: string
  time_stuck: number
  last_subsystem: string | null
  message_count: number
}

export function EventCorrelationDashboard() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedEvent, setSelectedEvent] = useState<CorrelatedEvent | null>(null)
  const [eventChain, setEventChain] = useState<CorrelatedEvent | null>(null)
  const [searchResults, setSearchResults] = useState<CorrelatedEvent[]>([])
  const [bottlenecks, setBottlenecks] = useState<EventBottleneck[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('search')
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const [filterType, setFilterType] = useState<string>('all')
  const [timeRange, setTimeRange] = useState<string>('24h')
  const [hasSearched, setHasSearched] = useState(false)

  // Mock data for demo
  const mockEventChain: CorrelatedEvent = {
    event_id: 'PROX-20250609-001',
    event_type: 'PROXIMITY',
    object_id: 'SAT-12345',
    creation_time: new Date().toISOString(),
    status: 'ACTIVE',
    message_count: 15,
    subsystem_flow: [
      'SS0:launch-detection:' + new Date(Date.now() - 3600000).toISOString(),
      'SS2:state-estimation:' + new Date(Date.now() - 1800000).toISOString(),
      'SS5:hostility-monitoring:' + new Date(Date.now() - 900000).toISOString()
    ],
    metadata: { threat_level: 'HIGH', range_km: 2.5 },
    parents: [],
    children: [
      {
        event_id: 'MAN-20250609-002',
        event_type: 'MANEUVER',
        object_id: 'SAT-12345',
        creation_time: new Date(Date.now() + 300000).toISOString(),
        status: 'PLANNED',
        message_count: 8,
        subsystem_flow: [
          'SS5:threat-response:' + new Date(Date.now() + 300000).toISOString(),
          'SS3:maneuver-planning:' + new Date(Date.now() + 600000).toISOString()
        ],
        metadata: { maneuver_type: 'AVOIDANCE', delta_v: 1.2 },
        parents: [],
        children: []
      }
    ]
  }

  // Additional mock events for search results
  const mockSearchResults: CorrelatedEvent[] = [
    mockEventChain,
    {
      event_id: 'LAUNCH-20250609-003',
      event_type: 'LAUNCH',
      object_id: 'SAT-67890',
      creation_time: new Date(Date.now() - 7200000).toISOString(),
      status: 'RESOLVED',
      message_count: 12,
      subsystem_flow: [
        'SS0:launch-detection:' + new Date(Date.now() - 7200000).toISOString(),
        'SS1:target-modeling:' + new Date(Date.now() - 6900000).toISOString(),
        'SS2:state-estimation:' + new Date(Date.now() - 6600000).toISOString()
      ],
      metadata: { launch_site: 'VAFB', threat_level: 'LOW' },
      parents: [],
      children: []
    },
    {
      event_id: 'SEP-20250609-004',
      event_type: 'SEPARATION',
      object_id: 'SAT-11111',
      creation_time: new Date(Date.now() - 3600000).toISOString(),
      status: 'ACTIVE',
      message_count: 6,
      subsystem_flow: [
        'SS2:state-estimation:' + new Date(Date.now() - 3600000).toISOString(),
        'SS5:hostility-monitoring:' + new Date(Date.now() - 3300000).toISOString()
      ],
      metadata: { separation_type: 'PLANNED', parent_object: 'RB-11111' },
      parents: [],
      children: []
    }
  ]

  const mockBottlenecks: EventBottleneck[] = [
    {
      event_id: 'MAN-20250609-007',
      object_id: 'SAT-67890',
      event_type: 'MANEUVER',
      creation_time: new Date(Date.now() - 1800000).toISOString(),
      time_stuck: 1800,
      last_subsystem: 'SS2',
      message_count: 8
    },
    {
      event_id: 'LAUNCH-20250609-003',
      object_id: 'SAT-11111',
      event_type: 'LAUNCH',
      creation_time: new Date(Date.now() - 3600000).toISOString(),
      time_stuck: 3600,
      last_subsystem: 'SS0',
      message_count: 3
    }
  ]

  // Fetch functions (using mock data for now)
  const fetchEventChain = useCallback(async (eventId: string) => {
    setLoading(true)
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      setEventChain(mockEventChain)
      setSelectedEvent(mockEventChain)
    } catch (error) {
      console.error('Failed to fetch event chain:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchBottlenecks = useCallback(async () => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 200))
      setBottlenecks(mockBottlenecks)
    } catch (error) {
      console.error('Failed to fetch bottlenecks:', error)
    }
  }, [])

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return
    
    setLoading(true)
    setHasSearched(true)
    try {
      // Simulate search
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Filter results based on search query
      let results = mockSearchResults
      if (searchQuery.toLowerCase().includes('prox')) {
        results = mockSearchResults.filter(e => e.event_type === 'PROXIMITY')
      } else if (searchQuery.toLowerCase().includes('launch')) {
        results = mockSearchResults.filter(e => e.event_type === 'LAUNCH')
      } else if (searchQuery.toLowerCase().includes('sat-12345')) {
        results = mockSearchResults.filter(e => e.object_id === 'SAT-12345')
      }
      
      setSearchResults(results)
      if (results.length > 0) {
        setEventChain(results[0])
        setSelectedEvent(results[0])
      }
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setLoading(false)
    }
  }, [searchQuery, mockSearchResults])

  const toggleNodeExpansion = (eventId: string) => {
    const newExpanded = new Set(expandedNodes)
    if (newExpanded.has(eventId)) {
      newExpanded.delete(eventId)
    } else {
      newExpanded.add(eventId)
    }
    setExpandedNodes(newExpanded)
  }

  useEffect(() => {
    fetchBottlenecks()
    
    // Load sample data if no search has been performed
    if (!hasSearched) {
      setTimeout(() => {
        setSearchResults(mockSearchResults)
        setEventChain(mockSearchResults[0])
        setSelectedEvent(mockSearchResults[0])
      }, 1000)
    }
    
    const interval = setInterval(fetchBottlenecks, 30000)
    return () => clearInterval(interval)
  }, [fetchBottlenecks, hasSearched, mockSearchResults])

  const getEventTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'LAUNCH': 'bg-red-500',
      'MAN': 'bg-blue-500',
      'PROX': 'bg-yellow-500',
      'SEP': 'bg-purple-500',
      'REENT': 'bg-orange-500',
      'ATT': 'bg-green-500',
      'LINK': 'bg-pink-500'
    }
    return colors[type] || 'bg-gray-500'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ACTIVE': return 'text-green-600'
      case 'RESOLVED': return 'text-blue-600'
      case 'ERROR': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const renderEventNode = (event: CorrelatedEvent, depth: number = 0) => {
    const isExpanded = expandedNodes.has(event.event_id)
    const hasChildren = event.children && event.children.length > 0
    const eventType = event.event_id.split('-')[0]
    
    return (
      <div key={event.event_id} style={{ marginLeft: `${depth * 20}px` }} className="my-2">
        <div 
          className="flex items-center gap-2 p-3 rounded-lg border border-gray-800 hover:bg-[#2A2F3E] cursor-pointer transition-colors"
          onClick={() => {
            setSelectedEvent(event)
            if (hasChildren) toggleNodeExpansion(event.event_id)
          }}
        >
          {hasChildren && (
            <Button
              variant="ghost"
              size="sm"
              className="p-0 h-6 w-6"
              onClick={(e) => {
                e.stopPropagation()
                toggleNodeExpansion(event.event_id)
              }}
            >
              {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </Button>
          )}
          
          <div className={`w-3 h-3 rounded-full ${getEventTypeColor(eventType)}`} />
          
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm">{event.event_id}</span>
              <Badge variant="outline" className="text-xs">
                {event.object_id}
              </Badge>
              <span className={`text-xs ${getStatusColor(event.status)}`}>
                {event.status}
              </span>
            </div>
            
            <div className="flex items-center gap-4 text-xs text-gray-500 mt-1">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDateSafe(event.creation_time, 'HH:mm:ss', 'N/A')}
              </span>
              <span className="flex items-center gap-1">
                <Layers className="h-3 w-3" />
                {event.message_count} messages
              </span>
              <span className="flex items-center gap-1">
                <GitBranch className="h-3 w-3" />
                {event.subsystem_flow.length} subsystems
              </span>
            </div>
          </div>
          
          {event.metadata?.threat_level && (
            <Badge 
              variant={event.metadata.threat_level === 'HIGH' ? 'destructive' : 'secondary'}
              className="text-xs"
            >
              {event.metadata.threat_level}
            </Badge>
          )}
        </div>
        
        {isExpanded && hasChildren && (
          <div className="mt-2">
            {event.children.map(child => renderEventNode(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex justify-end items-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchBottlenecks()}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
          >
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
      </div>

      {/* Bottleneck Alert */}
      {bottlenecks.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>{bottlenecks.length} events are experiencing delays.</strong>
            {' '}The oldest has been stuck for {Math.round(bottlenecks[0].time_stuck / 60)} minutes.
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="search">
            <Search className="h-4 w-4 mr-2" />
            Search
          </TabsTrigger>
          <TabsTrigger value="flow">
            <Network className="h-4 w-4 mr-2" />
            Flow Diagram
          </TabsTrigger>
          <TabsTrigger value="timeline">
            <Activity className="h-4 w-4 mr-2" />
            Timeline
          </TabsTrigger>
          <TabsTrigger value="bottlenecks">
            <Zap className="h-4 w-4 mr-2" />
            Bottlenecks
          </TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Event Search</CardTitle>
              <CardDescription>
                Search by Event ID (e.g., PROX-20250609-001) or Object ID
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter Event ID or Object ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  className="flex-1"
                />
                
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">All Types</option>
                  <option value="LAUNCH">Launch</option>
                  <option value="MAN">Maneuver</option>
                  <option value="PROX">Proximity</option>
                  <option value="SEP">Separation</option>
                </select>
                
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="1h">Last Hour</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </select>
                
                <Button onClick={handleSearch} disabled={loading}>
                  <Search className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>

          {(searchResults.length > 0 || !hasSearched) && (
            <Card>
              <CardHeader>
                <CardTitle>
                  {hasSearched ? `Search Results (${searchResults.length} events found)` : 'Recent Events (Sample Data)'}
                </CardTitle>
                <CardDescription>
                  {hasSearched ? 'Click on an event to view its correlation tree' : 'Sample events - try searching for "PROX", "LAUNCH", or "SAT-12345"'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {(searchResults.length > 0 ? searchResults : mockSearchResults).map((event) => (
                    <div
                      key={event.event_id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedEvent?.event_id === event.event_id ? 'bg-[#1E40AF] border-[#3B82F6]' : 'hover:bg-[#2A2F3E]'
                      }`}
                      onClick={() => {
                        setSelectedEvent(event)
                        setEventChain(event)
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full ${getEventTypeColor(event.event_id.split('-')[0])}`} />
                          <span className="font-mono text-sm font-medium">{event.event_id}</span>
                          <Badge variant="outline" className="text-xs">
                            {event.object_id}
                          </Badge>
                        </div>
                        <Badge className={getStatusColor(event.status)}>
                          {event.status}
                        </Badge>
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        {formatDateSafe(event.creation_time, 'PPp', 'Invalid date')} • {event.message_count} messages
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {hasSearched && searchResults.length === 0 && !loading && (
            <Card>
              <CardContent className="text-center py-8">
                <Search className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                <p className="text-gray-600">No events found matching your search</p>
                <p className="text-sm text-gray-500 mt-1">Try searching for "PROX", "LAUNCH", or "SAT-12345"</p>
              </CardContent>
            </Card>
          )}

          {eventChain && (
            <Card>
              <CardHeader>
                <CardTitle>Event Correlation Tree</CardTitle>
                <CardDescription>
                  Hierarchical view of correlated events
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[500px]">
                  {renderEventNode(eventChain)}
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {selectedEvent && (
            <Card>
              <CardHeader>
                <CardTitle>Event Details: {selectedEvent.event_id}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Object ID</p>
                    <p className="font-mono">{selectedEvent.object_id}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Status</p>
                    <Badge className={getStatusColor(selectedEvent.status)}>
                      {selectedEvent.status}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Created</p>
                    <p>{formatDateSafe(selectedEvent.creation_time, 'PPpp', 'Invalid date')}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Messages</p>
                    <p>{selectedEvent.message_count}</p>
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium text-gray-500 mb-2">Subsystem Flow</p>
                  <div className="space-y-1">
                    {selectedEvent.subsystem_flow.map((flow, idx) => {
                      const [subsystem, messageType, timestamp] = flow.split(':')
                      return (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <Badge variant="outline" className="text-xs">
                            {subsystem}
                          </Badge>
                          <span className="text-gray-600">{messageType}</span>
                          <span className="text-gray-400 text-xs">
                            {formatDateSafe(timestamp, 'HH:mm:ss.SSS', 'N/A')}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {Object.keys(selectedEvent.metadata).length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-500 mb-2">Metadata</p>
                    <pre className="text-xs bg-[#0A0E1A] border border-gray-700 text-gray-300 p-2 rounded overflow-auto">
                      {JSON.stringify(selectedEvent.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="flow">
          {eventChain ? (
            <Card>
              <CardHeader>
                <CardTitle>Event Flow Visualization</CardTitle>
                <CardDescription>
                  Interactive diagram showing event relationships and message flow
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <EventFlowDiagram event={eventChain} />
                  
                  {/* Message Flow Summary */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Message Flow</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          {eventChain.subsystem_flow.map((flow, idx) => {
                            const [subsystem, messageType] = flow.split(':')
                            return (
                              <div key={idx} className="flex items-center gap-2 text-sm">
                                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-xs font-medium">
                                  {idx + 1}
                                </div>
                                <div>
                                  <p className="font-medium">{subsystem}</p>
                                  <p className="text-gray-600 text-xs">{messageType}</p>
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Event Relationships</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div>
                            <p className="text-sm font-medium text-gray-500">Parent Events</p>
                            <p className="text-sm">{eventChain.parents.length === 0 ? 'None (root event)' : `${eventChain.parents.length} events`}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-500">Child Events</p>
                            <p className="text-sm">{eventChain.children.length === 0 ? 'None' : `${eventChain.children.length} events`}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-500">Total Messages</p>
                            <p className="text-sm">{eventChain.message_count} messages</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Network className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                <p className="text-gray-600">Select an event to view flow diagram</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="timeline">
          {eventChain ? (
            <Card>
              <CardHeader>
                <CardTitle>Event Timeline</CardTitle>
                <CardDescription>
                  Temporal view of event progression across subsystems
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <EventTimelineView event={eventChain} />
                  
                  {/* Timeline Details */}
                  <div className="border-l-2 border-blue-200 pl-4 space-y-4">
                    {eventChain.subsystem_flow.map((flow, idx) => {
                      const [subsystem, messageType, timestamp] = flow.split(':')
                      return (
                        <div key={idx} className="relative">
                          <div className="absolute -left-6 w-3 h-3 rounded-full bg-blue-500 border-2 border-white"></div>
                          <div className="bg-[#1A1F2E] border border-gray-800 rounded-lg p-3">
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="font-medium text-sm">{subsystem}</p>
                                <p className="text-gray-600 text-xs">{messageType}</p>
                              </div>
                              <Badge variant="outline" className="text-xs">
                                {formatDateSafe(timestamp, 'HH:mm:ss.SSS', 'N/A')}
                              </Badge>
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                              {formatDateSafe(timestamp, 'PPp', 'Invalid date')}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Activity className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                <p className="text-gray-600">Select an event to view timeline</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="bottlenecks">
          <Card>
            <CardHeader>
              <CardTitle>Event Processing Bottlenecks</CardTitle>
              <CardDescription>
                Events that are stuck or experiencing delays in the processing pipeline
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {bottlenecks.map((bottleneck) => (
                  <div
                    key={bottleneck.event_id}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                    onClick={() => fetchEventChain(bottleneck.event_id)}
                  >
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="h-5 w-5 text-orange-500" />
                      <div>
                        <p className="font-mono text-sm">{bottleneck.event_id}</p>
                        <p className="text-xs text-gray-500">
                          Object: {bottleneck.object_id} • Type: {bottleneck.event_type}
                        </p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <p className="text-sm font-medium text-red-600">
                        Stuck for {Math.round(bottleneck.time_stuck / 60)} minutes
                      </p>
                      <p className="text-xs text-gray-500">
                        Last seen: {bottleneck.last_subsystem || 'Unknown'}
                      </p>
                    </div>
                  </div>
                ))}
                
                {bottlenecks.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-500" />
                    <p>No processing bottlenecks detected</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 