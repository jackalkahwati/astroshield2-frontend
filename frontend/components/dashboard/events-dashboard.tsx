import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertTitle, AlertDescription } from '../ui/alert';
import { Loader2 } from 'lucide-react';

interface Event {
  id: string;
  event_type: string;
  object_id: string;
  status: string;
  creation_time: string;
  update_time: string;
  threat_level: string | null;
  coa_recommendation: {
    title: string;
    description: string;
    priority: number;
    actions: string[];
  } | null;
}

interface DashboardData {
  total_events: number;
  events_by_type: Record<string, number>;
  events_by_status: Record<string, number>;
  events_by_threat: Record<string, number>;
  recent_high_threats: Event[];
}

export default function EventsDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/v1/events/dashboard');
        
        if (!response.ok) {
          throw new Error('Failed to fetch dashboard data');
        }
        
        const data = await response.json();
        setDashboardData(data);
        
        // Also fetch recent events
        const eventsResponse = await fetch('/api/v1/events/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ limit: 10 }),
        });
        
        if (!eventsResponse.ok) {
          throw new Error('Failed to fetch events');
        }
        
        const eventsData = await eventsResponse.json();
        setEvents(eventsData);
        
        setLoading(false);
      } catch (err: any) {
        setError(err?.message || 'An error occurred');
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, []);

  // Function to get color based on threat level
  const getThreatColor = (threat: string | null) => {
    if (!threat) return 'bg-gray-200 text-gray-800';
    
    switch(threat.toLowerCase()) {
      case 'high':
        return 'bg-red-500 text-white';
      case 'moderate':
        return 'bg-orange-500 text-white';
      case 'low':
        return 'bg-yellow-500 text-black';
      default:
        return 'bg-blue-200 text-blue-800';
    }
  };

  // Function to get color based on status
  const getStatusColor = (status: string) => {
    switch(status.toLowerCase()) {
      case 'completed':
        return 'bg-green-500 text-white';
      case 'processing':
        return 'bg-blue-500 text-white';
      case 'error':
        return 'bg-red-500 text-white';
      case 'detected':
        return 'bg-purple-500 text-white';
      case 'awaiting_data':
        return 'bg-yellow-500 text-black';
      case 'rejected':
        return 'bg-gray-500 text-white';
      default:
        return 'bg-gray-200 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-gray-500" />
        <span className="ml-2">Loading event data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Event Overview</CardTitle>
            <CardDescription>Summary of all event processing</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData?.total_events || 0}</div>
            <p className="text-sm text-gray-500">Total Events</p>
            
            <div className="mt-4">
              <h4 className="font-semibold text-sm mb-1">By Type</h4>
              <div className="flex flex-wrap gap-2">
                {dashboardData?.events_by_type && Object.entries(dashboardData.events_by_type).map(([type, count]) => (
                  <Badge key={type} variant="outline">
                    {type}: {count}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Processing Status</CardTitle>
            <CardDescription>Current event processing status</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.events_by_status && Object.entries(dashboardData.events_by_status).map(([status, count]) => (
                <div key={status} className="flex justify-between items-center">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full ${getStatusColor(status)}`} />
                    <span className="ml-2 capitalize">{status.replace('_', ' ')}</span>
                  </div>
                  <span className="font-semibold">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Threat Assessment</CardTitle>
            <CardDescription>Distribution of threat levels</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.events_by_threat && Object.entries(dashboardData.events_by_threat).map(([threat, count]) => (
                <div key={threat} className="flex justify-between items-center">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full ${getThreatColor(threat)}`} />
                    <span className="ml-2 capitalize">{threat || 'Unknown'}</span>
                  </div>
                  <span className="font-semibold">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="recent">
        <div className="flex justify-between items-center mb-4">
          <TabsList>
            <TabsTrigger value="recent">Recent Events</TabsTrigger>
            <TabsTrigger value="threats">High Threats</TabsTrigger>
          </TabsList>
          
          <Button size="sm" variant="outline">
            Process Pending Events
          </Button>
        </div>
        
        <TabsContent value="recent" className="mt-0">
          <Card>
            <CardContent className="p-0">
              <div className="divide-y">
                {events.length > 0 ? (
                  events.map(event => (
                    <div 
                      key={event.id}
                      className="p-4 hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedEvent(event)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-semibold">
                            {event.event_type.replace('_', ' ')} - {event.object_id}
                          </h4>
                          <p className="text-sm text-gray-500">
                            {new Date(event.creation_time).toLocaleString()}
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <Badge className={getStatusColor(event.status)}>
                            {event.status.replace('_', ' ')}
                          </Badge>
                          {event.threat_level && (
                            <Badge className={getThreatColor(event.threat_level)}>
                              {event.threat_level}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-gray-500">No events found</div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="threats" className="mt-0">
          <Card>
            <CardContent className="p-0">
              <div className="divide-y">
                {dashboardData?.recent_high_threats && dashboardData.recent_high_threats.length > 0 ? (
                  dashboardData.recent_high_threats.map(event => (
                    <div 
                      key={event.id}
                      className="p-4 hover:bg-gray-50 cursor-pointer"
                      onClick={() => setSelectedEvent(event)}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-semibold">
                            {event.event_type.replace('_', ' ')} - {event.object_id}
                          </h4>
                          <p className="text-sm text-gray-500">
                            {new Date(event.creation_time).toLocaleString()}
                          </p>
                        </div>
                        <Badge className={getThreatColor(event.threat_level)}>
                          {event.threat_level}
                        </Badge>
                      </div>
                      {event.coa_recommendation && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-sm">
                          <p className="font-semibold">{event.coa_recommendation.title}</p>
                          <p className="text-gray-600 text-xs">{event.coa_recommendation.description}</p>
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-gray-500">No high threat events found</div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      {selectedEvent && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>{selectedEvent.event_type.replace('_', ' ')} Event Details</CardTitle>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setSelectedEvent(null)}
              >
                Close
              </Button>
            </div>
            <CardDescription>
              {selectedEvent.object_id} - Created {new Date(selectedEvent.creation_time).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex gap-2">
                <Badge className={getStatusColor(selectedEvent.status)}>
                  {selectedEvent.status.replace('_', ' ')}
                </Badge>
                {selectedEvent.threat_level && (
                  <Badge className={getThreatColor(selectedEvent.threat_level)}>
                    {selectedEvent.threat_level}
                  </Badge>
                )}
              </div>
              
              {selectedEvent.coa_recommendation && (
                <div className="border rounded-md p-4">
                  <h4 className="font-semibold text-lg mb-2">
                    {selectedEvent.coa_recommendation.title}
                  </h4>
                  <p className="text-gray-600 mb-4">
                    {selectedEvent.coa_recommendation.description}
                  </p>
                  <div>
                    <h5 className="font-semibold mb-2">Recommended Actions:</h5>
                    <ul className="list-disc pl-5 space-y-1">
                      {selectedEvent.coa_recommendation.actions.map((action, index) => (
                        <li key={index}>{action}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex justify-end">
            <Button variant="outline" size="sm">View Full Details</Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}