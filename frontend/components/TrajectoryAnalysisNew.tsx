import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle, Rocket, ArrowRight } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import TrajectoryViewer from './TrajectoryViewer';

interface TrajectoryConfig {
  object_name: string;
  object_properties: {
    mass: number;
    area: number;
    cd: number;
  };
  atmospheric_model: string;
  wind_model: string;
  monte_carlo_samples: number;
  breakup_model: {
    enabled: boolean;
    fragmentation_threshold: number;
  };
}

interface ImpactPrediction {
  time: number;
  position: [number, number, number];
  confidence: number;
  energy: number;
  area: number;
}

interface BreakupPoint {
  time: number;
  position: [number, number, number];
  fragments: number;
  cause: string;
}

interface TrajectoryPoint {
  time: number;
  position: [number, number, number];
  velocity: [number, number, number];
}

interface TrajectoryResult {
  trajectory: TrajectoryPoint[];
  impactPrediction: ImpactPrediction;
  breakupPoints: BreakupPoint[];
}

// Default configuration for trajectory analysis
const defaultConfig: TrajectoryConfig = {
  object_name: 'Satellite Debris',
  object_properties: {
    mass: 100,
    area: 1.2,
    cd: 2.2
  },
  atmospheric_model: 'exponential',
  wind_model: 'custom',
  monte_carlo_samples: 100,
  breakup_model: {
    enabled: true,
    fragmentation_threshold: 50
  }
};

// Mock data for preview
const generateMockData = (): TrajectoryResult => ({
  trajectory: Array(100).fill(0).map((_, i) => ({
    time: Date.now() / 1000 + i * 60,
    position: [
      -90 + Math.sin(i * 0.1) * 20,
      30 + Math.cos(i * 0.1) * 15,
      100000 - i * 1000
    ] as [number, number, number],
    velocity: [
      Math.sin(i * 0.1) * 200,
      Math.cos(i * 0.1) * 200,
      -100
    ] as [number, number, number]
  })),
  impactPrediction: {
    time: Date.now() / 1000 + 99 * 60,
    position: [
      -90 + Math.sin(99 * 0.1) * 20,
      30 + Math.cos(99 * 0.1) * 15,
      0
    ] as [number, number, number],
    confidence: 0.95,
    energy: 2500,
    area: 0.75
  },
  breakupPoints: [
    {
      time: Date.now() / 1000 + 50 * 60,
      position: [
        -90 + Math.sin(50 * 0.1) * 20,
        30 + Math.cos(50 * 0.1) * 15,
        50000
      ] as [number, number, number],
      fragments: 24,
      cause: 'Aerodynamic Stress'
    }
  ]
});

const TrajectoryAnalysisNew: React.FC = () => {
  const [config, setConfig] = useState<TrajectoryConfig>(defaultConfig);
  const [initialState, setInitialState] = useState<[number, number, number, number, number, number]>([0, 0, 400000, 7800, 0, 0]);
  const [trajectoryData, setTrajectoryData] = useState<TrajectoryResult | null>(generateMockData()); // Start with mock data
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('results'); // Start on results tab with mock data

  const handleChange = (field: string, value: any) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setConfig(prev => ({
        ...prev,
        [parent]: {
          ...prev[parent as keyof TrajectoryConfig],
          [child]: value
        }
      }));
    } else {
      setConfig(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const handleNestedChange = (parent: string, field: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [parent]: {
        ...prev[parent as keyof TrajectoryConfig],
        [field]: value
      }
    }));
  };

  const handleInitialStateChange = (index: number, value: number) => {
    setInitialState(prev => {
      const newState = [...prev] as [number, number, number, number, number, number];
      newState[index] = value;
      return newState;
    });
  };

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Always use mock data for now
      setTrajectoryData(generateMockData());
      setActiveTab('results');
    } catch (err) {
      console.error('Error analyzing trajectory:', err);
      setTrajectoryData(generateMockData());
      setActiveTab('results');
      setError('Could not connect to analysis server. Using mock data instead.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            Trajectory Analysis
          </CardTitle>
          <CardDescription>
            Analyze the trajectory of space objects including satellites, debris, and re-entering vehicles.
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
                        <Label htmlFor="object_name">Object Name</Label>
                        <Input 
                          id="object_name"
                          value={config.object_name}
                          onChange={(e) => handleChange('object_name', e.target.value)}
                          placeholder="Enter object name"
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="mass">Mass (kg)</Label>
                        <div className="flex gap-4 items-center">
                          <Slider 
                            id="mass"
                            value={[config.object_properties.mass]}
                            min={10}
                            max={10000}
                            step={10}
                            onValueChange={(values) => handleNestedChange('object_properties', 'mass', values[0])}
                            className="flex-1"
                          />
                          <Input 
                            type="number"
                            className="w-24"
                            value={config.object_properties.mass}
                            onChange={(e) => handleNestedChange('object_properties', 'mass', parseFloat(e.target.value))}
                          />
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="area">Cross-sectional Area (m²)</Label>
                        <div className="flex gap-4 items-center">
                          <Slider 
                            id="area"
                            value={[config.object_properties.area]}
                            min={0.1}
                            max={50}
                            step={0.1}
                            onValueChange={(values) => handleNestedChange('object_properties', 'area', values[0])}
                            className="flex-1"
                          />
                          <Input 
                            type="number"
                            className="w-24"
                            value={config.object_properties.area}
                            onChange={(e) => handleNestedChange('object_properties', 'area', parseFloat(e.target.value))}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mb-2">Model Settings</h3>
                    
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="atmospheric_model">Atmospheric Model</Label>
                        <Select 
                          value={config.atmospheric_model}
                          onValueChange={(value) => handleChange('atmospheric_model', value)}
                        >
                          <SelectTrigger id="atmospheric_model">
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
                        <Label htmlFor="wind_model">Wind Model</Label>
                        <Select 
                          value={config.wind_model}
                          onValueChange={(value) => handleChange('wind_model', value)}
                        >
                          <SelectTrigger id="wind_model">
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
                          value={initialState[0]}
                          onChange={(e) => handleInitialStateChange(0, parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="latitude">Latitude (°)</Label>
                        <Input 
                          id="latitude"
                          type="number"
                          value={initialState[1]}
                          onChange={(e) => handleInitialStateChange(1, parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="altitude">Altitude (m)</Label>
                        <Input 
                          id="altitude"
                          type="number"
                          value={initialState[2]}
                          onChange={(e) => handleInitialStateChange(2, parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <Label htmlFor="velocity_x">Velocity X (m/s)</Label>
                        <Input 
                          id="velocity_x"
                          type="number"
                          value={initialState[3]}
                          onChange={(e) => handleInitialStateChange(3, parseFloat(e.target.value))}
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
              {trajectoryData ? (
                <TrajectoryViewer 
                  trajectory={trajectoryData.trajectory}
                  impactPrediction={trajectoryData.impactPrediction}
                  breakupPoints={trajectoryData.breakupPoints}
                  autoPlay={true}
                />
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-500">No trajectory data available. Configure and analyze first.</p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default TrajectoryAnalysisNew; 