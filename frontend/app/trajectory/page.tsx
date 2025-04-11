"use client"

import { useState, useEffect, useRef } from "react"
import {
  Home,
  BarChart2,
  Bell,
  Settings,
  Menu,
  Layers,
  Calendar,
  Activity,
  Navigation,
  Compass,
  Zap,
  Sliders,
  Moon,
  Sun,
  Info,
  ChevronRight,
  Play,
  Pause,
  SkipForward,
  SkipBack,
} from "lucide-react"
import TrajectoryMap from "@/components/TrajectoryMap"
import { trajectoryApi } from "@/lib/api-client"
import { useToast } from "@/components/ui/use-toast"
import { TrajectoryConfig as TrajectoryConfigType, TrajectoryRequest, TrajectoryResult } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

export default function SpacecraftTrajectoryAnalysis() {
  const { toast } = useToast()
  const [darkMode, setDarkMode] = useState(true)
  const [showUncertainty, setShowUncertainty] = useState(true)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [timeIndex, setTimeIndex] = useState(75)
  const animationRef = useRef<number | null>(null)
  
  // Form state
  const [config, setConfig] = useState({
    atmospheric_model: "NRLMSISE-00",
    wind_model: "HWM14",
    monte_carlo_samples: 1000,
    object_properties: {
      mass: 1000,
      area: 1.0,
      cd: 2.2
    },
    breakup_model: {
      enabled: false,
      fragment_count: 10,
      mass_distribution: "log_normal",
      velocity_perturbation: 100.0
    }
  })
  
  const [initialState, setInitialState] = useState({
    position: { x: 0, y: 0, z: 6471000 },
    velocity: { x: 1000, y: 0, z: -1000 }
  })
  
  // API state
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [trajectoryData, setTrajectoryData] = useState<any>(null)
  
  // Handle playback animation
  useEffect(() => {
    if (isPlaying) {
      const playbackSpeed = 1 // Frames per second
      
      const animate = () => {
        setTimeIndex(prev => {
          if (prev >= 100) {
            setIsPlaying(false)
            return 100
          }
          return prev + 1
        })
        
        if (isPlaying) {
          animationRef.current = requestAnimationFrame(animate)
        }
      }
      
      animationRef.current = requestAnimationFrame(animate)
      
      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }
  }, [isPlaying])
  
  const togglePlayback = () => setIsPlaying(!isPlaying)
  const jumpBackward = () => setTimeIndex(prev => Math.max(0, prev - 10))
  const jumpForward = () => setTimeIndex(prev => Math.min(100, prev + 10))
  
  // Validate form fields
  const validateForm = (): boolean => {
    // Check object properties have positive values
    if (config.object_properties.mass <= 0) {
      toast({
        title: "Validation Error",
        description: "Mass must be greater than zero",
        variant: "destructive",
      })
      return false
    }
    
    if (config.object_properties.area <= 0) {
      toast({
        title: "Validation Error",
        description: "Area must be greater than zero",
        variant: "destructive",
      })
      return false
    }
    
    if (config.object_properties.cd <= 0) {
      toast({
        title: "Validation Error",
        description: "Drag coefficient must be greater than zero",
        variant: "destructive",
      })
      return false
    }
    
    // Validate Monte Carlo samples (must be positive integer)
    if (config.monte_carlo_samples < 1 || !Number.isInteger(config.monte_carlo_samples)) {
      toast({
        title: "Validation Error",
        description: "Monte Carlo samples must be a positive integer",
        variant: "destructive",
      })
      return false
    }
    
    // If breakup is enabled, validate breakup model parameters
    if (config.breakup_model.enabled) {
      if (config.breakup_model.fragment_count < 1 || !Number.isInteger(config.breakup_model.fragment_count)) {
        toast({
          title: "Validation Error",
          description: "Fragment count must be a positive integer",
          variant: "destructive",
        })
        return false
      }
      
      if (config.breakup_model.velocity_perturbation <= 0) {
        toast({
          title: "Validation Error",
          description: "Velocity perturbation must be greater than zero",
          variant: "destructive",
        })
        return false
      }
    }
    
    return true
  }

  // Function to run trajectory analysis
  const runAnalysis = async () => {
    // Validate form first
    if (!validateForm()) {
      return
    }
    
    setIsLoading(true)
    setError(null)
    
    try {
      // Convert form values to API request
      const request: TrajectoryRequest = {
        config: {
          atmospheric_model: config.atmospheric_model,
          wind_model: config.wind_model,
          monte_carlo_samples: config.monte_carlo_samples,
          object_properties: config.object_properties,
          breakup_model: config.breakup_model
        },
        initial_state: [
          initialState.position.x,
          initialState.position.y,
          initialState.position.z,
          initialState.velocity.x,
          initialState.velocity.y,
          initialState.velocity.z
        ]
      }
      
      // Call API
      const response = await trajectoryApi.analyzeTrajectory(request)
      
      if (response.data) {
        setTrajectoryData(response.data)
        toast({
          title: "Analysis Complete",
          description: "Trajectory analysis has been completed successfully.",
        })
      } else if (response.error) {
        setError(response.error.message)
        toast({
          title: "Analysis Failed",
          description: response.error.message,
          variant: "destructive",
        })
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100">
        {/* Main Content */}
        <div className="flex flex-col min-h-screen">
          {/* Page Content */}
          <div className="flex-1 overflow-auto bg-slate-50 dark:bg-slate-950">
            <div className="container mx-auto p-6 max-w-7xl">
              {/* Page Header */}
              <div className="flex justify-between items-center mb-8 pt-4">
                <div>
                  <h1 className="text-3xl font-bold tracking-tight mb-1 text-slate-900 dark:text-white">
                    Spacecraft Trajectory Analysis
                  </h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Advanced predictive modeling for spacecraft trajectory analysis, reentry path simulation, and impact
                    risk assessment with high-precision atmospheric modeling.
                  </p>
                </div>
                
                {/* Theme toggle */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-9 w-9 rounded-full border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300"
                        onClick={() => setDarkMode(!darkMode)}
                      >
                        {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Toggle theme</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Configuration Panel */}
                <div className="lg:col-span-1 space-y-6">
                  <Card className="border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-slate-900 dark:text-white">Configuration</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <Accordion type="single" collapsible defaultValue="item-1" className="w-full">
                        <AccordionItem value="item-1" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Atmospheric Model
                          </AccordionTrigger>
                          <AccordionContent>
                            <Select 
                              value={config.atmospheric_model}
                              onValueChange={(value) => setConfig({...config, atmospheric_model: value})}
                            >
                              <SelectTrigger className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white">
                                <SelectValue placeholder="Select model" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="NRLMSISE-00">NRLMSISE-00</SelectItem>
                                <SelectItem value="JB2008">JB2008</SelectItem>
                                <SelectItem value="MSIS-E-90">MSIS-E-90</SelectItem>
                              </SelectContent>
                            </Select>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="item-2" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Wind Model
                          </AccordionTrigger>
                          <AccordionContent>
                            <Select 
                              value={config.wind_model}
                              onValueChange={(value) => setConfig({...config, wind_model: value})}
                            >
                              <SelectTrigger className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white">
                                <SelectValue placeholder="Select model" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="HWM14">HWM14</SelectItem>
                                <SelectItem value="HWM07">HWM07</SelectItem>
                                <SelectItem value="Custom">Custom</SelectItem>
                              </SelectContent>
                            </Select>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="item-3" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Monte Carlo Samples
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="flex items-center space-x-2">
                              <Input
                                type="number"
                                value={config.monte_carlo_samples}
                                onChange={(e) => setConfig({...config, monte_carlo_samples: parseInt(e.target.value) || 1000})}
                                className="w-full border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                              />
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="h-8 w-8 text-slate-600 dark:text-slate-400"
                                    >
                                      <Info className="h-4 w-4" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>Number of Monte Carlo simulation runs</TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="item-4" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Object Properties
                          </AccordionTrigger>
                          <AccordionContent className="space-y-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <label className="text-sm font-medium text-slate-900 dark:text-white">Mass (kg)</label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6 text-slate-600 dark:text-slate-400"
                                      >
                                        <Info className="h-3 w-3" />
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Object mass in kilograms</TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                              <Input
                                type="number"
                                value={config.object_properties.mass}
                                onChange={(e) => setConfig({
                                  ...config, 
                                  object_properties: {
                                    ...config.object_properties,
                                    mass: parseFloat(e.target.value) || 0
                                  }
                                })}
                                className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <label className="text-sm font-medium text-slate-900 dark:text-white">Area (m²)</label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6 text-slate-600 dark:text-slate-400"
                                      >
                                        <Info className="h-3 w-3" />
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Cross-sectional area in square meters</TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                              <Input
                                type="number"
                                value={config.object_properties.area}
                                onChange={(e) => setConfig({
                                  ...config, 
                                  object_properties: {
                                    ...config.object_properties,
                                    area: parseFloat(e.target.value) || 0
                                  }
                                })}
                                className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                              />
                            </div>

                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <label className="text-sm font-medium text-slate-900 dark:text-white">
                                  Drag Coefficient
                                </label>
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6 text-slate-600 dark:text-slate-400"
                                      >
                                        <Info className="h-3 w-3" />
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Dimensionless drag coefficient</TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                              <Input
                                type="number"
                                value={config.object_properties.cd}
                                onChange={(e) => setConfig({
                                  ...config, 
                                  object_properties: {
                                    ...config.object_properties,
                                    cd: parseFloat(e.target.value) || 0
                                  }
                                })}
                                step="0.1"
                                className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                              />
                            </div>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="item-5" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Breakup Model
                          </AccordionTrigger>
                          <AccordionContent>
                            <Select 
                              value={config.breakup_model.enabled ? "simple" : "no"}
                              onValueChange={(value) => setConfig({
                                ...config,
                                breakup_model: {
                                  ...config.breakup_model,
                                  enabled: value !== "no"
                                }
                              })}
                            >
                              <SelectTrigger className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white">
                                <SelectValue placeholder="Select model" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="no">No</SelectItem>
                                <SelectItem value="simple">Simple Breakup</SelectItem>
                                <SelectItem value="advanced">Advanced Breakup</SelectItem>
                              </SelectContent>
                            </Select>
                          </AccordionContent>
                        </AccordionItem>

                        <AccordionItem value="item-6" className="border-slate-200 dark:border-slate-800">
                          <AccordionTrigger className="py-2 text-slate-900 dark:text-white">
                            Initial State
                          </AccordionTrigger>
                          <AccordionContent className="space-y-4">
                            <div className="space-y-2">
                              <label className="text-sm font-medium text-slate-900 dark:text-white">Position (m)</label>
                              <div className="grid grid-cols-3 gap-2">
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">X</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.position.x}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      position: {
                                        ...initialState.position,
                                        x: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Y</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.position.y}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      position: {
                                        ...initialState.position,
                                        y: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Z</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.position.z}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      position: {
                                        ...initialState.position,
                                        z: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                              </div>
                            </div>

                            <div className="space-y-2">
                              <label className="text-sm font-medium text-slate-900 dark:text-white">
                                Velocity (m/s)
                              </label>
                              <div className="grid grid-cols-3 gap-2">
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">X</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.velocity.x}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      velocity: {
                                        ...initialState.velocity,
                                        x: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Y</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.velocity.y}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      velocity: {
                                        ...initialState.velocity,
                                        y: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                                <div className="space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500 dark:text-slate-400">Z</span>
                                  </div>
                                  <Input
                                    type="number"
                                    value={initialState.velocity.z}
                                    onChange={(e) => setInitialState({
                                      ...initialState,
                                      velocity: {
                                        ...initialState.velocity,
                                        z: parseFloat(e.target.value) || 0
                                      }
                                    })}
                                    className="border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                                  />
                                </div>
                              </div>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </Accordion>
                    </CardContent>
                    <CardFooter className="flex flex-col gap-3">
                      <Button 
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                        onClick={runAnalysis}
                        disabled={isLoading}
                      >
                        {isLoading ? 'Processing...' : 'Run Analysis'}
                      </Button>
                      <Button 
                        variant="outline" 
                        className="w-full text-slate-600 dark:text-slate-400 border-slate-300 dark:border-slate-700"
                        onClick={() => {
                          // Reset to defaults
                          setConfig({
                            atmospheric_model: "NRLMSISE-00",
                            wind_model: "HWM14",
                            monte_carlo_samples: 1000,
                            object_properties: {
                              mass: 1000,
                              area: 1.0,
                              cd: 2.2
                            },
                            breakup_model: {
                              enabled: false,
                              fragment_count: 10,
                              mass_distribution: "log_normal",
                              velocity_perturbation: 100.0
                            }
                          });
                          
                          setInitialState({
                            position: { x: 0, y: 0, z: 6471000 },
                            velocity: { x: 1000, y: 0, z: -1000 }
                          });
                        }}
                        disabled={isLoading}
                      >
                        Reset to Defaults
                      </Button>
                    </CardFooter>
                  </Card>
                </div>

                {/* Visualization and Results */}
                <div className="lg:col-span-2 space-y-6">
                  <Card className="border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-slate-900 dark:text-white">Trajectory Visualization</CardTitle>
                        <div className="flex items-center gap-4">
                          <div className="flex items-center space-x-2">
                            <Switch id="uncertainty" checked={showUncertainty} onCheckedChange={setShowUncertainty} />
                            <label htmlFor="uncertainty" className="text-sm font-medium text-slate-900 dark:text-white">
                              Show Uncertainty
                            </label>
                          </div>
                          <Select defaultValue="3d">
                            <SelectTrigger className="w-[120px] border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white">
                              <SelectValue placeholder="View Mode" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="3d">3D View</SelectItem>
                              <SelectItem value="2d">2D View</SelectItem>
                              <SelectItem value="satellite">Satellite View</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-0 relative">
                      <div className="aspect-video relative overflow-hidden rounded-md">
                        <TrajectoryMap 
                          showUncertainty={showUncertainty}
                          darkMode={darkMode}
                          timeIndex={timeIndex}
                          maxTimeIndex={100}
                          trajectoryData={trajectoryData}
                          isLoading={isLoading}
                        />
                      </div>
                      <div className="p-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-slate-900 dark:text-white">
                              Trajectory Timeline
                            </span>
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {new Date(new Date('2025-03-24T05:00:00.000Z').getTime() + (timeIndex * 600)).toISOString().replace('T', ' ').substring(0, 19)}
                            </span>
                          </div>
                          <Slider 
                            value={[timeIndex]} 
                            max={100} 
                            step={1}
                            onValueChange={(value) => setTimeIndex(value[0])}
                          />
                          <div className="flex items-center justify-center gap-2 pt-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 rounded-full text-slate-700 dark:text-slate-300"
                              onClick={jumpBackward}
                            >
                              <SkipBack className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="default"
                              size="sm"
                              className="h-9 w-9 p-0 rounded-full bg-blue-600 hover:bg-blue-700"
                              onClick={togglePlayback}
                            >
                              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 rounded-full text-slate-700 dark:text-slate-300"
                              onClick={jumpForward}
                            >
                              <SkipForward className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Impact Prediction */}
                    <Card className="border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-slate-900 dark:text-white">Impact Prediction</CardTitle>
                          <Badge
                            variant="outline"
                            className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800/30"
                          >
                            95.0% Confidence
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <p className="text-sm text-slate-500 dark:text-slate-400">Time</p>
                              <p className="font-medium text-slate-900 dark:text-white">
                                {new Date(new Date('2025-03-24T05:00:00.000Z').getTime() + (timeIndex * 600)).toISOString().replace('T', ' ').substring(0, 19)}
                              </p>
                            </div>
                            <div>
                              <p className="text-sm text-slate-500 dark:text-slate-400">Velocity</p>
                              <p className="font-medium text-slate-900 dark:text-white">1000.00 m/s</p>
                            </div>
                          </div>

                          <Separator className="bg-slate-200 dark:bg-slate-800" />

                          <div>
                            <p className="text-sm text-slate-500 dark:text-slate-400">Location</p>
                            <div className="flex items-center gap-2">
                              <p className="font-medium text-slate-900 dark:text-white">41.7128°N, 73.0060°E</p>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-slate-600 dark:text-slate-400"
                              >
                                <Info className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>

                          <Separator className="bg-slate-200 dark:bg-slate-800" />

                          <div>
                            <p className="text-sm text-slate-500 dark:text-slate-400">Uncertainty</p>
                            <p className="font-medium text-slate-900 dark:text-white">±10.0 km</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Breakup Events */}
                    <Card className="border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-slate-900 dark:text-white">Breakup Events</CardTitle>
                          <Badge
                            variant="outline"
                            className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800/30"
                          >
                            15 Fragments
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <Tabs defaultValue="event1">
                          <TabsList className="grid w-full grid-cols-2 bg-slate-100 dark:bg-slate-800">
                            <TabsTrigger
                              value="event1"
                              className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-900 data-[state=active]:text-slate-900 dark:data-[state=active]:text-white"
                            >
                              Event 1
                            </TabsTrigger>
                            <TabsTrigger
                              value="event2"
                              className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-900 data-[state=active]:text-slate-900 dark:data-[state=active]:text-white"
                            >
                              Event 2
                            </TabsTrigger>
                          </TabsList>
                          <TabsContent value="event1" className="space-y-4 pt-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Time</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">
                                  {new Date(new Date('2025-03-24T04:50:00.000Z').getTime() + (timeIndex * 600)).toISOString().replace('T', ' ').substring(0, 19)}
                                </p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Altitude</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">80000.0 km</p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Fragments</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">5</p>
                              </div>
                            </div>
                          </TabsContent>
                          <TabsContent value="event2" className="space-y-4 pt-3">
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Time</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">
                                  {new Date(new Date('2025-03-24T04:55:00.000Z').getTime() + (timeIndex * 600)).toISOString().replace('T', ' ').substring(0, 19)}
                                </p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Altitude</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">60000.0 km</p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-slate-500 dark:text-slate-400">Fragments</p>
                                <p className="text-sm font-medium text-slate-900 dark:text-white">10</p>
                              </div>
                            </div>
                          </TabsContent>
                        </Tabs>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}