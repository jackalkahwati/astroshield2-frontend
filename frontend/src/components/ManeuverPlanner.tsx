import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Rocket, Calendar, AlertTriangle, CheckCircle2, Clock } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { format } from 'date-fns';

interface ManeuverType {
  id: string;
  name: string;
  description: string;
}

interface SatelliteOption {
  id: string;
  name: string;
  type: string;
  orbit: string;
}

interface ManeuverParameters {
  delta_v: number;
  burn_duration: number;
  direction: {
    x: number;
    y: number;
    z: number;
  };
  target_orbit?: {
    altitude: number;
    inclination: number;
    eccentricity: number;
  };
}

interface ManeuverResource {
  fuel_remaining: number;
  power_available: number;
  thruster_status: string;
}

interface ManeuverRequest {
  satellite_id: string;
  type: string;
  scheduled_time: string;
  parameters: ManeuverParameters;
  priority: number;
  notes?: string;
}

interface ManeuverStatus {
  id: string;
  satellite_id: string;
  status: string;
  type: string;
  start_time: string;
  end_time?: string;
  resources: ManeuverResource;
  parameters: ManeuverParameters;
  created_by?: string;
  created_at: string;
  updated_at?: string;
}

interface SimulationResult {
  success: boolean;
  satellite_id: string;
  type: string;
  fuel_required: number;
  expected_results: {
    collision_probability_change: number;
    orbit_stability_change: number;
    estimated_completion_time: string;
  };
}

// Mock data for UI demonstration
const maneuverTypes: ManeuverType[] = [
  { id: 'collision_avoidance', name: 'Collision Avoidance', description: 'Avoid potential collision with another space object' },
  { id: 'station_keeping', name: 'Station Keeping', description: 'Maintain desired orbital position' },
  { id: 'debris_avoidance', name: 'Debris Avoidance', description: 'Avoid debris field or specific debris object' },
  { id: 'orbit_raising', name: 'Orbit Raising', description: 'Increase orbital altitude' },
  { id: 'orbit_lowering', name: 'Orbit Lowering', description: 'Decrease orbital altitude' },
  { id: 'inclination_change', name: 'Inclination Change', description: 'Change orbital inclination' },
  { id: 'deorbit', name: 'Deorbit', description: 'Controlled reentry to Earth' }
];

const satelliteOptions: SatelliteOption[] = [
  { id: 'sat-001', name: 'AstroShield-1', type: 'Surveillance', orbit: 'LEO' },
  { id: 'sat-002', name: 'AstroShield-2', type: 'Communications', orbit: 'GEO' },
  { id: 'sat-003', name: 'Sentinel-A', type: 'Observation', orbit: 'SSO' },
  { id: 'sat-004', name: 'Guardian-1', type: 'Defense', orbit: 'MEO' }
];

const mockManeuvers: ManeuverStatus[] = [
  {
    id: 'mnv-001',
    satellite_id: 'sat-001',
    status: 'completed',
    type: 'collision_avoidance',
    start_time: new Date(Date.now() - 2 * 3600 * 1000).toISOString(),
    end_time: new Date(Date.now() - 1.75 * 3600 * 1000).toISOString(),
    resources: {
      fuel_remaining: 85.5,
      power_available: 90.0,
      thruster_status: 'nominal'
    },
    parameters: {
      delta_v: 0.02,
      burn_duration: 15.0,
      direction: { x: 0.1, y: 0.0, z: -0.1 },
      target_orbit: { altitude: 500.2, inclination: 45.0, eccentricity: 0.001 }
    },
    created_by: 'system',
    created_at: new Date(Date.now() - 24 * 3600 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1.75 * 3600 * 1000).toISOString()
  },
  {
    id: 'mnv-002',
    satellite_id: 'sat-001',
    status: 'scheduled',
    type: 'station_keeping',
    start_time: new Date(Date.now() + 5 * 3600 * 1000).toISOString(),
    resources: {
      fuel_remaining: 85.5,
      power_available: 90.0,
      thruster_status: 'nominal'
    },
    parameters: {
      delta_v: 0.01,
      burn_duration: 10.0,
      direction: { x: 0.0, y: 0.0, z: 0.1 },
      target_orbit: { altitude: 500.0, inclination: 45.0, eccentricity: 0.001 }
    },
    created_by: 'system',
    created_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString()
  }
];

const defaultManeuverRequest: ManeuverRequest = {
  satellite_id: '',
  type: '',
  scheduled_time: new Date().toISOString(),
  parameters: {
    delta_v: 0.01,
    burn_duration: 10.0,
    direction: { x: 0.0, y: 0.0, z: 0.1 }
  },
  priority: 1
};

const ManeuverPlanner: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('active');
  const [maneuvers, setManeuvers] = useState<ManeuverStatus[]>(mockManeuvers);
  const [maneuverRequest, setManeuverRequest] = useState<ManeuverRequest>(defaultManeuverRequest);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [resources, setResources] = useState<ManeuverResource | null>(null);

  // Fetch active maneuvers and satellite resources on component mount
  useEffect(() => {
    const fetchData = async () => {
      // In a real app, these would be API calls
      // For now, we'll use the mock data
      setManeuvers(mockManeuvers);
      
      if (maneuverRequest.satellite_id) {
        setResources({
          fuel_remaining: 85.5,
          power_available: 90.0,
          thruster_status: 'nominal'
        });
      }
    };
    
    fetchData();
  }, [maneuverRequest.satellite_id]);

  const handleChange = (field: string, value: any) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      if (parent === 'parameters') {
        setManeuverRequest(prev => ({
          ...prev,
          parameters: {
            ...prev.parameters,
            [child]: value
          }
        }));
      } else if (parent === 'direction') {
        setManeuverRequest(prev => ({
          ...prev,
          parameters: {
            ...prev.parameters,
            direction: {
              ...prev.parameters.direction,
              [child]: value
            }
          }
        }));
      } else if (parent === 'target_orbit') {
        setManeuverRequest(prev => ({
          ...prev,
          parameters: {
            ...prev.parameters,
            target_orbit: {
              ...(prev.parameters.target_orbit || { altitude: 0, inclination: 0, eccentricity: 0 }),
              [child]: value
            }
          }
        }));
      }
    } else {
      setManeuverRequest(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const handleSatelliteChange = (satelliteId: string) => {
    handleChange('satellite_id', satelliteId);
    // Reset simulation results when satellite changes
    setSimulationResult(null);
  };

  const handleSimulate = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // For now, simulate a delay and return mock data
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setSimulationResult({
        success: true,
        satellite_id: maneuverRequest.satellite_id,
        type: maneuverRequest.type,
        fuel_required: maneuverRequest.parameters.delta_v * 10,
        expected_results: {
          collision_probability_change: maneuverRequest.type === 'collision_avoidance' ? -0.95 : 0.0,
          orbit_stability_change: maneuverRequest.type === 'station_keeping' ? 0.25 : -0.1,
          estimated_completion_time: new Date(
            new Date(maneuverRequest.scheduled_time).getTime() + 
            maneuverRequest.parameters.burn_duration * 1000
          ).toISOString()
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // For now, simulate a delay and add to local state
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const newManeuver: ManeuverStatus = {
        id: `mnv-${Date.now().toString().slice(-6)}`,
        satellite_id: maneuverRequest.satellite_id,
        status: 'scheduled',
        type: maneuverRequest.type,
        start_time: maneuverRequest.scheduled_time,
        resources: resources || {
          fuel_remaining: 85.5,
          power_available: 90.0,
          thruster_status: 'nominal'
        },
        parameters: maneuverRequest.parameters,
        created_by: 'current-user',
        created_at: new Date().toISOString()
      };
      
      setManeuvers(prev => [newManeuver, ...prev]);
      setManeuverRequest(defaultManeuverRequest);
      setSimulationResult(null);
      setActiveTab('active');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = async (maneuverID: string) => {
    setIsLoading(true);
    
    try {
      // In a real app, this would be an API call
      // For now, simulate a delay and update local state
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setManeuvers(prev => 
        prev.map(m => 
          m.id === maneuverID 
            ? { ...m, status: 'canceled', updated_at: new Date().toISOString() } 
            : m
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'scheduled':
        return <Badge variant="outline" className="flex items-center gap-1"><Calendar className="h-3 w-3" /> Scheduled</Badge>;
      case 'in_progress':
        return <Badge variant="secondary" className="flex items-center gap-1"><Clock className="h-3 w-3" /> In Progress</Badge>;
      case 'completed':
        return <Badge variant="success" className="flex items-center gap-1"><CheckCircle2 className="h-3 w-3" /> Completed</Badge>;
      case 'canceled':
        return <Badge variant="destructive" className="flex items-center gap-1"><AlertTriangle className="h-3 w-3" /> Canceled</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  const getManeuverTypeName = (typeId: string): string => {
    const type = maneuverTypes.find(t => t.id === typeId);
    return type ? type.name : typeId;
  };

  const getSatelliteName = (satelliteId: string): string => {
    const satellite = satelliteOptions.find(s => s.id === satelliteId);
    return satellite ? satellite.name : satelliteId;
  };

  return (
    <div className="container mx-auto p-4">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            Maneuver Planning
          </CardTitle>
          <CardDescription>
            Plan, schedule, and monitor spacecraft maneuvers for collision avoidance and orbit adjustments.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="active">Active Maneuvers</TabsTrigger>
              <TabsTrigger value="history">Maneuver History</TabsTrigger>
              <TabsTrigger value="create">Create Maneuver</TabsTrigger>
            </TabsList>
            
            <TabsContent value="active" className="space-y-4">
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Satellite</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Scheduled Time</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {maneuvers
                      .filter(m => ['scheduled', 'in_progress'].includes(m.status))
                      .map(maneuver => (
                        <TableRow key={maneuver.id}>
                          <TableCell className="font-mono">{maneuver.id}</TableCell>
                          <TableCell>{getSatelliteName(maneuver.satellite_id)}</TableCell>
                          <TableCell>{getManeuverTypeName(maneuver.type)}</TableCell>
                          <TableCell>{getStatusBadge(maneuver.status)}</TableCell>
                          <TableCell>{format(new Date(maneuver.start_time), 'MMM dd, yyyy HH:mm')}</TableCell>
                          <TableCell>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              disabled={isLoading || maneuver.status === 'in_progress'} 
                              onClick={() => handleCancel(maneuver.id)}
                            >
                              Cancel
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                      
                    {maneuvers.filter(m => ['scheduled', 'in_progress'].includes(m.status)).length === 0 && (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-4">
                          No active maneuvers scheduled
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </TabsContent>
            
            <TabsContent value="history" className="space-y-4">
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Satellite</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Execution Time</TableHead>
                      <TableHead>ΔV (m/s)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {maneuvers
                      .filter(m => ['completed', 'canceled'].includes(m.status))
                      .map(maneuver => (
                        <TableRow key={maneuver.id}>
                          <TableCell className="font-mono">{maneuver.id}</TableCell>
                          <TableCell>{getSatelliteName(maneuver.satellite_id)}</TableCell>
                          <TableCell>{getManeuverTypeName(maneuver.type)}</TableCell>
                          <TableCell>{getStatusBadge(maneuver.status)}</TableCell>
                          <TableCell>
                            {format(new Date(maneuver.start_time), 'MMM dd, yyyy HH:mm')}
                            {maneuver.end_time && ` (${Math.round((new Date(maneuver.end_time).getTime() - new Date(maneuver.start_time).getTime()) / 1000 / 60)} min)`}
                          </TableCell>
                          <TableCell>{maneuver.parameters.delta_v.toFixed(4)}</TableCell>
                        </TableRow>
                      ))}
                      
                    {maneuvers.filter(m => ['completed', 'canceled'].includes(m.status)).length === 0 && (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-4">
                          No maneuver history available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </TabsContent>
            
            <TabsContent value="create" className="space-y-4">
              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Maneuver Details</h3>
                    
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <Label htmlFor="satellite_id">Satellite</Label>
                        <Select 
                          value={maneuverRequest.satellite_id}
                          onValueChange={handleSatelliteChange}
                        >
                          <SelectTrigger id="satellite_id">
                            <SelectValue placeholder="Select a satellite" />
                          </SelectTrigger>
                          <SelectContent>
                            {satelliteOptions.map(sat => (
                              <SelectItem key={sat.id} value={sat.id}>
                                {sat.name} ({sat.orbit})
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      
                      {resources && (
                        <div className="rounded-md bg-muted p-3 space-y-2">
                          <h4 className="text-sm font-medium">Satellite Resources</h4>
                          <div className="space-y-1">
                            <div className="text-xs text-muted-foreground">Fuel Remaining</div>
                            <Progress value={resources.fuel_remaining} className="h-2" />
                            <div className="text-xs text-right">{resources.fuel_remaining.toFixed(1)}%</div>
                          </div>
                          <div className="space-y-1">
                            <div className="text-xs text-muted-foreground">Power Available</div>
                            <Progress value={resources.power_available} className="h-2" />
                            <div className="text-xs text-right">{resources.power_available.toFixed(1)}%</div>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-xs text-muted-foreground">Thruster Status:</span>
                            <Badge 
                              variant={resources.thruster_status === 'nominal' ? 'outline' : 'destructive'}
                              className="text-xs"
                            >
                              {resources.thruster_status}
                            </Badge>
                          </div>
                        </div>
                      )}
                      
                      <div className="space-y-1">
                        <Label htmlFor="maneuver_type">Maneuver Type</Label>
                        <Select 
                          value={maneuverRequest.type}
                          onValueChange={(value) => handleChange('type', value)}
                        >
                          <SelectTrigger id="maneuver_type">
                            <SelectValue placeholder="Select maneuver type" />
                          </SelectTrigger>
                          <SelectContent>
                            {maneuverTypes.map(type => (
                              <SelectItem key={type.id} value={type.id}>
                                {type.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        {maneuverRequest.type && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {maneuverTypes.find(t => t.id === maneuverRequest.type)?.description}
                          </p>
                        )}
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="scheduled_time">Scheduled Time</Label>
                        <Input 
                          id="scheduled_time"
                          type="datetime-local"
                          value={new Date(maneuverRequest.scheduled_time).toISOString().slice(0, 16)}
                          onChange={(e) => handleChange('scheduled_time', new Date(e.target.value).toISOString())}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="delta_v">Delta-V (m/s)</Label>
                        <Input 
                          id="delta_v"
                          type="number"
                          step="0.001"
                          min="0"
                          value={maneuverRequest.parameters.delta_v}
                          onChange={(e) => handleChange('parameters.delta_v', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="burn_duration">Burn Duration (seconds)</Label>
                        <Input 
                          id="burn_duration"
                          type="number"
                          step="0.1"
                          min="0"
                          value={maneuverRequest.parameters.burn_duration}
                          onChange={(e) => handleChange('parameters.burn_duration', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-md font-medium mb-2">Direction Vector</h4>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="space-y-1">
                        <Label htmlFor="direction_x">X</Label>
                        <Input 
                          id="direction_x"
                          type="number"
                          step="0.01"
                          value={maneuverRequest.parameters.direction.x}
                          onChange={(e) => handleChange('direction.x', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="direction_y">Y</Label>
                        <Input 
                          id="direction_y"
                          type="number"
                          step="0.01"
                          value={maneuverRequest.parameters.direction.y}
                          onChange={(e) => handleChange('direction.y', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="direction_z">Z</Label>
                        <Input 
                          id="direction_z"
                          type="number"
                          step="0.01"
                          value={maneuverRequest.parameters.direction.z}
                          onChange={(e) => handleChange('direction.z', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Target Orbit Parameters</h3>
                    
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <Label htmlFor="target_altitude">Target Altitude (km)</Label>
                        <Input 
                          id="target_altitude"
                          type="number"
                          step="0.1"
                          value={maneuverRequest.parameters.target_orbit?.altitude || ''}
                          onChange={(e) => handleChange('target_orbit.altitude', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="target_inclination">Target Inclination (°)</Label>
                        <Input 
                          id="target_inclination"
                          type="number"
                          step="0.1"
                          min="0"
                          max="180"
                          value={maneuverRequest.parameters.target_orbit?.inclination || ''}
                          onChange={(e) => handleChange('target_orbit.inclination', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="target_eccentricity">Target Eccentricity</Label>
                        <Input 
                          id="target_eccentricity"
                          type="number"
                          step="0.0001"
                          min="0"
                          max="1"
                          value={maneuverRequest.parameters.target_orbit?.eccentricity || ''}
                          onChange={(e) => handleChange('target_orbit.eccentricity', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mb-2">Notes</h3>
                    <div className="space-y-1">
                      <Label htmlFor="notes">Additional Notes</Label>
                      <textarea 
                        id="notes"
                        className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        placeholder="Enter any additional notes or context for this maneuver"
                        value={maneuverRequest.notes || ''}
                        onChange={(e) => handleChange('notes', e.target.value)}
                      />
                    </div>
                  </div>
                  
                  {simulationResult && (
                    <div className="rounded-md bg-muted p-3 space-y-2">
                      <h4 className="text-sm font-medium">Simulation Results</h4>
                      
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Fuel Required:</span>
                          <span className="text-xs font-medium">{simulationResult.fuel_required.toFixed(2)}%</span>
                        </div>
                        
                        {simulationResult.expected_results.collision_probability_change !== 0 && (
                          <div className="flex justify-between">
                            <span className="text-xs text-muted-foreground">Collision Probability Change:</span>
                            <span className={`text-xs font-medium ${simulationResult.expected_results.collision_probability_change < 0 ? 'text-green-500' : 'text-red-500'}`}>
                              {(simulationResult.expected_results.collision_probability_change * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                        
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Orbit Stability Change:</span>
                          <span className={`text-xs font-medium ${simulationResult.expected_results.orbit_stability_change > 0 ? 'text-green-500' : 'text-yellow-500'}`}>
                            {(simulationResult.expected_results.orbit_stability_change * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Completion Time:</span>
                          <span className="text-xs font-medium">
                            {format(new Date(simulationResult.expected_results.estimated_completion_time), 'MMM dd, yyyy HH:mm:ss')}
                          </span>
                        </div>
                      </div>
                      
                      {resources && simulationResult.fuel_required > resources.fuel_remaining && (
                        <Alert variant="destructive" className="py-2">
                          <AlertTitle className="text-xs">Insufficient Fuel</AlertTitle>
                          <AlertDescription className="text-xs">
                            This maneuver requires more fuel than available ({simulationResult.fuel_required.toFixed(1)}% vs {resources.fuel_remaining.toFixed(1)}%)
                          </AlertDescription>
                        </Alert>
                      )}
                    </div>
                  )}
                  
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      className="flex-1"
                      onClick={handleSimulate}
                      disabled={isLoading || !maneuverRequest.satellite_id || !maneuverRequest.type}
                    >
                      {isLoading && !simulationResult ? 'Simulating...' : 'Simulate Maneuver'}
                    </Button>
                    
                    <Button 
                      className="flex-1"
                      onClick={handleCreate}
                      disabled={
                        isLoading || 
                        !maneuverRequest.satellite_id || 
                        !maneuverRequest.type || 
                        !simulationResult ||
                        (resources && simulationResult && simulationResult.fuel_required > resources.fuel_remaining)
                      }
                    >
                      {isLoading && simulationResult ? 'Creating...' : 'Create Maneuver'}
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default ManeuverPlanner; 