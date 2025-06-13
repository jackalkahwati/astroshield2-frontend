"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Eye, EyeOff, Maximize2, Minimize2, RotateCw } from "lucide-react"
import { StatusIndicator, mapStatusType } from "@/components/ui/status-indicator"
import { Badge } from "@/components/ui/badge"
import { LoadingSpinner } from "@/components/ui/loading-spinner"

interface OrbitVisualizationProps {
  className?: string
  satellites?: SatelliteData[]
  isLoading?: boolean
}

interface SatelliteData {
  id: string
  name: string
  status: string
  orbitType: string
  inclination: number
  altitude: number
  position: {
    x: number
    y: number
    z: number
  }
}

// Sample data
const sampleSatellites: SatelliteData[] = [
  {
    id: "SAT-001",
    name: "Reconnaissance-1",
    status: "Active",
    orbitType: "LEO",
    inclination: 51.6,
    altitude: 420,
    position: { x: 0.2, y: 0.7, z: 0.1 }
  },
  {
    id: "SAT-002",
    name: "Communications-A",
    status: "Warning",
    orbitType: "GEO",
    inclination: 0,
    altitude: 35786,
    position: { x: 0.8, y: 0.3, z: 0.5 }
  },
  {
    id: "SAT-003",
    name: "Weather-Sat-3",
    status: "Active",
    orbitType: "SSO",
    inclination: 98.1,
    altitude: 705,
    position: { x: 0.5, y: 0.5, z: 0.8 }
  },
  {
    id: "SAT-004",
    name: "ISS-Monitor",
    status: "Critical",
    orbitType: "LEO",
    inclination: 51.6,
    altitude: 420,
    position: { x: 0.3, y: 0.2, z: 0.6 }
  }
]

export function SatelliteOrbitVisualization({ 
  className,
  satellites = sampleSatellites,
  isLoading = false
}: OrbitVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [viewMode, setViewMode] = useState<"3d" | "2d">("3d")
  const [rotationSpeed, setRotationSpeed] = useState<number>(50)
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false)
  const [selectedOrbitType, setSelectedOrbitType] = useState<string>("all")
  const [showLabels, setShowLabels] = useState<boolean>(true)
  const [rotation, setRotation] = useState<number>(0)
  const [selectedSatellite, setSelectedSatellite] = useState<SatelliteData | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Filter satellites based on selected orbit type
  const filteredSatellites = selectedOrbitType === "all" 
    ? satellites 
    : satellites.filter(sat => sat.orbitType === selectedOrbitType)

  // Animation effect
  useEffect(() => {
    if (isLoading) return

    let animationId: number
    const animate = () => {
      setRotation(prev => (prev + rotationSpeed / 1000) % 360)
      animationId = requestAnimationFrame(animate)
    }
    
    animationId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationId)
  }, [rotationSpeed, isLoading])

  // Canvas rendering effect
  useEffect(() => {
    if (isLoading || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas dimensions
    const updateCanvasSize = () => {
      const container = containerRef.current
      if (!container) return
      
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }
    
    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw Earth
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const earthRadius = Math.min(centerX, centerY) * 0.2
    
    // Earth gradient
    const earthGradient = ctx.createRadialGradient(
      centerX, centerY, 0, 
      centerX, centerY, earthRadius
    )
    earthGradient.addColorStop(0, '#1d4ed8')
    earthGradient.addColorStop(1, '#3b82f6')
    
    ctx.beginPath()
    ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2)
    ctx.fillStyle = earthGradient
    ctx.fill()
    
    // Add a subtle glow effect to Earth
    ctx.beginPath()
    ctx.arc(centerX, centerY, earthRadius + 5, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.fill()
    
    // Draw orbit paths for each satellite
    filteredSatellites.forEach((satellite, index) => {
      const orbitRadius = earthRadius * (1 + satellite.altitude / 10000)
      
      // Draw orbit path
      ctx.beginPath()
      ctx.ellipse(
        centerX, 
        centerY, 
        orbitRadius, 
        orbitRadius * (viewMode === "3d" ? 0.6 : 1), // Flatter for 3D effect
        rotation / 10 + index * 0.2, // Rotate orbit based on global rotation
        0, 
        Math.PI * 2
      )
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
      ctx.stroke()
      
      // Calculate satellite position on orbit
      const angle = rotation + index * (Math.PI / 2)
      const satX = centerX + orbitRadius * Math.cos(angle)
      const satY = centerY + orbitRadius * Math.sin(angle) * (viewMode === "3d" ? 0.6 : 1)
      
      // Draw satellite
      ctx.beginPath()
      ctx.arc(satX, satY, 5, 0, Math.PI * 2)
      
      // Color based on status
      let satColor = '#22c55e' // Active - Green
      if (satellite.status === 'Warning') satColor = '#eab308' // Warning - Yellow
      if (satellite.status === 'Critical') satColor = '#ef4444' // Critical - Red
      
      ctx.fillStyle = satColor
      ctx.fill()
      
      // Draw satellite glow effect
      ctx.beginPath()
      ctx.arc(satX, satY, 8, 0, Math.PI * 2)
      ctx.fillStyle = `${satColor}40` // Add transparency
      ctx.fill()
      
      // Draw label if enabled
      if (showLabels) {
        ctx.font = '12px Arial'
        ctx.fillStyle = '#ffffff'
        ctx.textAlign = 'center'
        ctx.fillText(satellite.name, satX, satY - 15)
      }
      
      // Highlight selected satellite
      if (selectedSatellite?.id === satellite.id) {
        ctx.beginPath()
        ctx.arc(satX, satY, 12, 0, Math.PI * 2)
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.lineWidth = 1
      }
    })
    
    return () => {
      window.removeEventListener('resize', updateCanvasSize)
    }
  }, [rotation, filteredSatellites, viewMode, showLabels, selectedSatellite, isLoading])

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!containerRef.current) return
    
    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
    }
  }

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Satellite Orbit Visualization</CardTitle>
          <CardDescription>Loading orbital data...</CardDescription>
        </CardHeader>
        <CardContent className="h-[400px] flex items-center justify-center">
          <LoadingSpinner size="lg" text="Loading visualization..." />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle>Satellite Orbit Visualization</CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowLabels(!showLabels)}
              title={showLabels ? "Hide labels" : "Show labels"}
            >
              {showLabels ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={toggleFullscreen}
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        <CardDescription>Real-time satellite orbital positions</CardDescription>
      </CardHeader>
      <CardContent className="p-0 relative">
        <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
          <Select value={selectedOrbitType} onValueChange={setSelectedOrbitType}>
            <SelectTrigger className="w-[120px] bg-background/80 backdrop-blur-sm">
              <SelectValue placeholder="Orbit Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Orbits</SelectItem>
              <SelectItem value="LEO">LEO</SelectItem>
              <SelectItem value="GEO">GEO</SelectItem>
              <SelectItem value="SSO">SSO</SelectItem>
              <SelectItem value="MEO">MEO</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={viewMode} onValueChange={(value) => setViewMode(value as "3d" | "2d")}>
            <SelectTrigger className="w-[120px] bg-background/80 backdrop-blur-sm">
              <SelectValue placeholder="View Mode" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="3d">3D View</SelectItem>
              <SelectItem value="2d">2D View</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="bg-background/80 backdrop-blur-sm p-2 rounded-md flex items-center gap-2">
            <RotateCw className="h-4 w-4 text-muted-foreground" />
            <Slider 
              value={[rotationSpeed]} 
              onValueChange={values => setRotationSpeed(values[0])}
              max={100}
              step={1}
              className="w-20"
            />
          </div>
        </div>

        <div 
          ref={containerRef} 
          className="relative h-[400px] overflow-hidden bg-slate-950 rounded-md"
        >
          <canvas 
            ref={canvasRef} 
            className="w-full h-full"
          />
        </div>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 pt-4">
        <div className="text-sm font-medium">Satellite Status</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 w-full">
          {satellites.slice(0, 4).map(satellite => (
            <Button
              key={satellite.id}
              variant="outline"
              className={`justify-start h-auto py-2 px-3 ${selectedSatellite?.id === satellite.id ? 'border-primary' : ''}`}
              onClick={() => setSelectedSatellite(selectedSatellite?.id === satellite.id ? null : satellite)}
            >
              <div className="flex flex-col items-start gap-1 text-left">
                <div className="flex items-center gap-2">
                  <StatusIndicator status={mapStatusType(satellite.status)} />
                  <span className="font-medium text-xs">{satellite.name}</span>
                </div>
                <div className="flex gap-1">
                  <Badge variant="outline" className="text-xs h-5 px-1">
                    {satellite.orbitType}
                  </Badge>
                  <Badge variant="outline" className="text-xs h-5 px-1">
                    {satellite.altitude} km
                  </Badge>
                </div>
              </div>
            </Button>
          ))}
        </div>
      </CardFooter>
    </Card>
  )
} 