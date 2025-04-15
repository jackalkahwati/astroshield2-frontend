"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, Rocket, ArrowRight } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export default function TrajectoryPage() {
  // Default configuration
  const [config, setConfig] = useState({
    objectName: 'Satellite Debris',
    mass: 100,
    area: 1.2,
    dragCoefficient: 2.2,
    atmosphericModel: 'exponential',
    windModel: 'custom',
    monteCarloSamples: 100
  });

  const [initialState, setInitialState] = useState({
    longitude: 0,
    latitude: 0,
    altitude: 400000,
    velocityX: 7800,
    velocityY: 0,
    velocityZ: 0
  });

  const [activeTab, setActiveTab] = useState('config');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);

  const handleConfigChange = (field: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleStateChange = (field: string, value: any) => {
    setInitialState(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simulate analysis
      await new Promise(resolve => setTimeout(resolve, 1500));
      setShowResults(true);
      setActiveTab('results');
    } catch (err) {
      console.error('Error:', err);
      setError('An error occurred during analysis');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            Trajectory Analysis
          </CardTitle>
          <CardDescription>
            Analyze spacecraft trajectories, predict reentry paths, and assess impact risks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="config">Configure</TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>
            
            <TabsContent value="config" className="space-y-4">
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
                    <h3 className="text-lg font-medium mb-2">Object Properties</h3>
                    
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="objectName">Object Name</Label>
                        <Input 
                          id="objectName"
                          value={config.objectName}
                          onChange={(e) => handleConfigChange('objectName', e.target.value)}
                          placeholder="Enter object name"
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="mass">Mass (kg)</Label>
                        <div className="flex gap-4 items-center">
                          <Slider 
                            id="mass"
                            value={[config.mass]}
                            min={10}
                            max={10000}
                            step={10}
                            onValueChange={(values) => handleConfigChange('mass', values[0])}
                            className="flex-1"
                          />
                          <Input 
                            type="number"
                            className="w-24"
                            value={config.mass}
                            onChange={(e) => handleConfigChange('mass', parseFloat(e.target.value))}
                          />
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="area">Cross-sectional Area (m²)</Label>
                        <div className="flex gap-4 items-center">
                          <Slider 
                            id="area"
                            value={[config.area]}
                            min={0.1}
                            max={50}
                            step={0.1}
                            onValueChange={(values) => handleConfigChange('area', values[0])}
                            className="flex-1"
                          />
                          <Input 
                            type="number"
                            className="w-24"
                            value={config.area}
                            onChange={(e) => handleConfigChange('area', parseFloat(e.target.value))}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mb-2">Model Settings</h3>
                    
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="atmosphericModel">Atmospheric Model</Label>
                        <Select 
                          value={config.atmosphericModel}
                          onValueChange={(value) => handleConfigChange('atmosphericModel', value)}
                        >
                          <SelectTrigger id="atmosphericModel">
                            <SelectValue placeholder="Select an atmospheric model" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="exponential">Exponential Model</SelectItem>
                            <SelectItem value="jacchia">Jacchia-Roberts Model</SelectItem>
                            <SelectItem value="msis">NRLMSISE-00 Model</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="windModel">Wind Model</Label>
                        <Select 
                          value={config.windModel}
                          onValueChange={(value) => handleConfigChange('windModel', value)}
                        >
                          <SelectTrigger id="windModel">
                            <SelectValue placeholder="Select a wind model" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Wind</SelectItem>
                            <SelectItem value="constant">Constant Wind</SelectItem>
                            <SelectItem value="custom">Custom Wind Profile</SelectItem>
                            <SelectItem value="hwm">Horizontal Wind Model</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Initial State</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1">
                        <Label htmlFor="longitude">Longitude (°)</Label>
                        <Input 
                          id="longitude"
                          type="number"
                          value={initialState.longitude}
                          onChange={(e) => handleStateChange('longitude', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="latitude">Latitude (°)</Label>
                        <Input 
                          id="latitude"
                          type="number"
                          value={initialState.latitude}
                          onChange={(e) => handleStateChange('latitude', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="altitude">Altitude (m)</Label>
                        <Input 
                          id="altitude"
                          type="number"
                          value={initialState.altitude}
                          onChange={(e) => handleStateChange('altitude', parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="velocityX">Velocity X (m/s)</Label>
                        <Input 
                          id="velocityX"
                          type="number"
                          value={initialState.velocityX}
                          onChange={(e) => handleStateChange('velocityX', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="pt-4">
                    <Button 
                      className="w-full"
                      onClick={handleAnalyze}
                      disabled={isLoading}
                    >
                      {isLoading ? 'Analyzing...' : 'Analyze Trajectory'} <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                    
                    <p className="text-sm text-gray-500 mt-2 text-center">
                      This will analyze the trajectory based on the provided parameters
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="results">
              {showResults ? (
                <div className="space-y-6">
                  <div className="p-6 bg-muted rounded-lg">
                    <h3 className="text-xl font-bold mb-4">Trajectory Simulation Results</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <h4 className="text-lg font-semibold">Object Information</h4>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="font-medium">Name:</div>
                          <div>{config.objectName}</div>
                          <div className="font-medium">Mass:</div>
                          <div>{config.mass} kg</div>
                          <div className="font-medium">Area:</div>
                          <div>{config.area} m²</div>
                          <div className="font-medium">Initial Altitude:</div>
                          <div>{initialState.altitude} m</div>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <h4 className="text-lg font-semibold">Impact Prediction</h4>
                        <div className="p-4 bg-yellow-100 dark:bg-yellow-950 rounded-md">
                          <div className="grid grid-cols-2 gap-2">
                            <div className="font-medium">Impact Time:</div>
                            <div>{new Date(Date.now() + 6000000).toLocaleString()}</div>
                            <div className="font-medium">Latitude:</div>
                            <div>{(initialState.latitude + 5.25).toFixed(4)}°</div>
                            <div className="font-medium">Longitude:</div>
                            <div>{(initialState.longitude - 8.75).toFixed(4)}°</div>
                            <div className="font-medium">Impact Energy:</div>
                            <div>3,567 kJ</div>
                            <div className="font-medium">Confidence:</div>
                            <div>84%</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-8 p-6 border rounded-md bg-background">
                      <h4 className="text-lg font-semibold mb-4">Visualization Placeholder</h4>
                      <div className="aspect-video bg-gradient-to-br from-blue-950 to-indigo-900 rounded-md flex items-center justify-center">
                        <p className="text-white text-center p-4">
                          3D trajectory visualization not available in this preview.<br/>
                          Full visualization will be available in the next update.
                        </p>
                      </div>
                    </div>
                    
                    <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <h4 className="text-lg font-semibold">Key Events</h4>
                        <div className="space-y-2">
                          <div className="p-3 bg-blue-100 dark:bg-blue-950 rounded">
                            <div className="font-medium">Re-entry Interface</div>
                            <div className="text-sm">Altitude: 100 km at {new Date(Date.now() + 1800000).toLocaleString()}</div>
                          </div>
                          <div className="p-3 bg-orange-100 dark:bg-orange-950 rounded">
                            <div className="font-medium">Maximum Heating</div>
                            <div className="text-sm">Altitude: 67 km at {new Date(Date.now() + 2700000).toLocaleString()}</div>
                          </div>
                          <div className="p-3 bg-red-100 dark:bg-red-950 rounded">
                            <div className="font-medium">Breakup Event</div>
                            <div className="text-sm">Altitude: 42 km at {new Date(Date.now() + 3600000).toLocaleString()}</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <h4 className="text-lg font-semibold">Statistics</h4>
                        <div className="space-y-2">
                          <div className="p-3 bg-background border rounded">
                            <div className="font-medium">Peak Deceleration</div>
                            <div className="text-sm">18.3 G at altitude 52 km</div>
                          </div>
                          <div className="p-3 bg-background border rounded">
                            <div className="font-medium">Peak Dynamic Pressure</div>
                            <div className="text-sm">78.5 kPa at altitude 48 km</div>
                          </div>
                          <div className="p-3 bg-background border rounded">
                            <div className="font-medium">Debris Field</div>
                            <div className="text-sm">Radius: 3.2 km around impact point</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-between">
                    <Button variant="outline" onClick={() => setActiveTab('config')}>
                      Modify Parameters
                    </Button>
                    <Button>
                      Export Results
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="text-center py-16">
                  <p className="text-muted-foreground mb-4">No trajectory data available.</p>
                  <Button variant="outline" onClick={() => setActiveTab('config')}>
                    Configure and Analyze
                  </Button>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
} 