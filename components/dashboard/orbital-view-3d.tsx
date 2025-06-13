"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Globe, Satellite, AlertTriangle, Target, RefreshCw, Radar, RotateCcw } from 'lucide-react'
import { getThreatLevelColor, HEX_COLORS } from '@/lib/chart-colors'

interface SatelliteObject {
  id: string
  name: string
  position: { x: number; y: number; z: number }
  velocity: { x: number; y: number; z: number }
  type: 'active' | 'debris' | 'threat'
  altitude: number
  threat_level: 'none' | 'low' | 'medium' | 'high' | 'critical'
  lat: number
  lng: number
  orbitalPhase: number
}

interface TrackingData {
  totalObjects: number
  activeTracking: number
  threats: number
  primaryAsset: string
  coverage: string
}

export const OrbitalView3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [rotation, setRotation] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 })
  const [trackingData, setTrackingData] = useState<TrackingData>({
    totalObjects: 47,
    activeTracking: 45,
    threats: 2,
    primaryAsset: "USA-317 (ISS)",
    coverage: "Global"
  })

  const [satellites, setSatellites] = useState<SatelliteObject[]>([
    {
      id: "USA-317",
      name: "International Space Station",
      position: { x: 0, y: 0, z: 400 },
      velocity: { x: 7.66, y: 0, z: 0 },
      type: 'active',
      altitude: 408,
      threat_level: 'none',
      lat: 45.0,
      lng: -93.0,
      orbitalPhase: 0
    },
    {
      id: "STARLINK-4729", 
      name: "Starlink-4729",
      position: { x: 200, y: 150, z: 550 },
      velocity: { x: 7.5, y: 0.1, z: 0 },
      type: 'active',
      altitude: 547,
      threat_level: 'medium',
      lat: 52.5,
      lng: 13.4,
      orbitalPhase: 90
    },
    {
      id: "DEB-001",
      name: "Unknown Debris",
      position: { x: -100, y: 80, z: 380 },
      velocity: { x: 7.8, y: -0.2, z: 0.1 },
      type: 'debris',
      altitude: 375,
      threat_level: 'high',
      lat: 35.7,
      lng: 139.7,
      orbitalPhase: 180
    },
    {
      id: "COSMOS-1408-DEB",
      name: "Cosmos 1408 Fragment",
      position: { x: 300, y: -200, z: 500 },
      velocity: { x: 7.4, y: 0.3, z: -0.1 },
      type: 'debris', 
      altitude: 485,
      threat_level: 'critical',
      lat: -33.9,
      lng: 18.4,
      orbitalPhase: 270
    }
  ])

  // Convert lat/lng to 3D coordinates
  const latLngTo3D = (lat: number, lng: number, radius: number) => {
    const phi = (90 - lat) * (Math.PI / 180)
    const theta = (lng + 180) * (Math.PI / 180)

    return {
      x: -(radius * Math.sin(phi) * Math.cos(theta)),
      y: radius * Math.cos(phi),
      z: radius * Math.sin(phi) * Math.sin(theta)
    }
  }

  // Enhanced 3D Globe Canvas Renderer
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const earthRadius = 120

    const render = () => {
      // Clear canvas
      ctx.fillStyle = '#0A0E1A'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw stars
      ctx.fillStyle = '#FFFFFF'
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        const size = Math.random() * 2
        ctx.globalAlpha = Math.random() * 0.8 + 0.2
        ctx.fillRect(x, y, size, size)
      }
      ctx.globalAlpha = 1

      // Draw Earth with 3D effect
      const gradient = ctx.createRadialGradient(
        centerX - 30, centerY - 30, 0,
        centerX, centerY, earthRadius
      )
      gradient.addColorStop(0, '#4A90E2')
      gradient.addColorStop(0.3, '#2E5C96')
      gradient.addColorStop(0.7, '#1E3A5F')
      gradient.addColorStop(1, '#0F1B2E')

      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2)
      ctx.fill()

      // Add continents (simplified)
      ctx.fillStyle = '#2D5016'
      ctx.globalAlpha = 0.8

      // North America
      ctx.beginPath()
      ctx.ellipse(centerX - 40, centerY - 20, 25, 35, Math.PI / 6, 0, Math.PI * 2)
      ctx.fill()

      // Europe/Africa
      ctx.beginPath()
      ctx.ellipse(centerX + 10, centerY - 10, 15, 40, -Math.PI / 8, 0, Math.PI * 2)
      ctx.fill()

      // Asia
      ctx.beginPath()
      ctx.ellipse(centerX + 35, centerY - 15, 20, 30, Math.PI / 12, 0, Math.PI * 2)
      ctx.fill()

      ctx.globalAlpha = 1

      // Draw satellite orbital paths
      satellites.forEach((satellite, index) => {
        const orbitRadius = earthRadius + (satellite.altitude - 400) * 0.3 + 60
        
        ctx.strokeStyle = getThreatLevelColor(satellite.threat_level)
        ctx.globalAlpha = 0.3
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2)
        ctx.stroke()
        ctx.globalAlpha = 1
      })

      // Draw satellites
      satellites.forEach((satellite, index) => {
        const time = Date.now() * 0.001
        const orbitRadius = earthRadius + (satellite.altitude - 400) * 0.3 + 60
        const angle = (satellite.orbitalPhase * Math.PI / 180) + time * 0.5 + index * 0.5

        const satX = centerX + Math.cos(angle) * orbitRadius
        const satY = centerY + Math.sin(angle) * orbitRadius * 0.7 // Elliptical for 3D effect

        // Satellite glow
        const glowGradient = ctx.createRadialGradient(satX, satY, 0, satX, satY, 15)
        glowGradient.addColorStop(0, getThreatLevelColor(satellite.threat_level))
        glowGradient.addColorStop(1, 'transparent')
        
        ctx.fillStyle = glowGradient
        ctx.fillRect(satX - 15, satY - 15, 30, 30)

        // Satellite marker
        ctx.fillStyle = getThreatLevelColor(satellite.threat_level)
        ctx.beginPath()
        ctx.arc(satX, satY, 4, 0, Math.PI * 2)
        ctx.fill()

        // Satellite border
        ctx.strokeStyle = '#FFFFFF'
        ctx.lineWidth = 1
        ctx.stroke()

        // Satellite ID (for debugging)
        ctx.fillStyle = '#FFFFFF'
        ctx.font = '10px Arial'
        ctx.fillText(satellite.id, satX + 8, satY - 8)
      })

      // Add atmospheric glow
      const atmosphereGradient = ctx.createRadialGradient(
        centerX, centerY, earthRadius,
        centerX, centerY, earthRadius + 20
      )
      atmosphereGradient.addColorStop(0, 'rgba(135, 206, 235, 0.3)')
      atmosphereGradient.addColorStop(1, 'transparent')
      
      ctx.fillStyle = atmosphereGradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, earthRadius + 20, 0, Math.PI * 2)
      ctx.fill()
    }

    const interval = setInterval(render, 50) // 20 FPS for smooth animation
    render() // Initial render

    return () => clearInterval(interval)
  }, [satellites])

  // Mouse interaction for rotation
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return

    const deltaX = e.clientX - lastMouse.x
    const deltaY = e.clientY - lastMouse.y

    setRotation(prev => ({
      x: prev.x + deltaY * 0.5,
      y: prev.y + deltaX * 0.5
    }))

    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  // Update satellite positions
  useEffect(() => {
    const updateInterval = setInterval(() => {
      setSatellites(prevSats => 
        prevSats.map(sat => ({
          ...sat,
          orbitalPhase: (sat.orbitalPhase + 1) % 360,
          position: {
            x: sat.position.x + (Math.random() - 0.5) * 5,
            y: sat.position.y + (Math.random() - 0.5) * 5,
            z: sat.altitude + (Math.random() - 0.5) * 3
          }
        }))
      )
      
      setTrackingData(prev => ({
        ...prev,
        activeTracking: 45 + Math.floor(Math.random() * 3),
        threats: Math.floor(Math.random() * 4)
      }))
    }, 100) // Fast updates for smooth orbital motion

    return () => clearInterval(updateInterval)
  }, [])

  const getObjectColor = (satellite: SatelliteObject) => {
    return getThreatLevelColor(satellite.threat_level)
  }

  const getThreatVariant = (level: string) => {
    switch (level) {
      case 'critical': return 'destructive'
      case 'high': return 'destructive' 
      case 'medium': return 'secondary'
      case 'low': return 'outline'
      default: return 'default'
    }
  }

  const handleRefreshTracking = () => {
    setTrackingData(prev => ({
      ...prev,
      totalObjects: 45 + Math.floor(Math.random() * 5),
      activeTracking: 43 + Math.floor(Math.random() * 5),
      threats: Math.floor(Math.random() * 5)
    }))
    
    setSatellites(prevSats => 
      prevSats.map(sat => {
        const threatLevels = ['none', 'low', 'medium', 'high', 'critical']
        const randomThreat = threatLevels[Math.floor(Math.random() * threatLevels.length)]
        return {
          ...sat,
          threat_level: randomThreat as any,
          orbitalPhase: Math.random() * 360
        }
      })
    )
  }

  const handleFocusPrimary = () => {
    // Reset rotation to focus on primary asset
    setRotation({ x: 0, y: 0 })
  }

  const handleResetView = () => {
    setRotation({ x: 0, y: 0 })
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-white">
            <Globe className="h-5 w-5" />
            Orbital Situational Awareness
          </CardTitle>
          <Badge variant="outline" className="text-xs bg-[#2A2F3E] text-white">
            Enhanced 3D View
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* 3D Globe View */}
          <div className="lg:col-span-2">
            <div className="h-96 border-2 border-gray-700 rounded relative overflow-hidden bg-[#0A0E1A]"
                 style={{ borderColor: HEX_COLORS.border }}>
              <canvas
                ref={canvasRef}
                width={800}
                height={384}
                className="w-full h-full cursor-grab active:cursor-grabbing"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
                  transformStyle: 'preserve-3d'
                }}
              />
              
              {/* Status overlay */}
              <div className="absolute top-4 right-4 border rounded p-3"
                   style={{ 
                     backgroundColor: HEX_COLORS.background.secondary,
                     borderColor: HEX_COLORS.border
                   }}>
                <div className="text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <Radar className="h-3 w-3" style={{ color: HEX_COLORS.status.good }} />
                    <span style={{ color: HEX_COLORS.status.good }}>TRACKING ACTIVE</span>
                  </div>
                  <div className="text-white">Coverage: {trackingData.coverage}</div>
                  <div className="text-xs text-gray-400">3D Globe: Canvas</div>
                </div>
              </div>

              {/* Interactive hint */}
              <div className="absolute bottom-4 left-4 text-xs text-gray-400">
                <div className="flex items-center gap-1">
                  <RotateCcw className="h-3 w-3" />
                  Drag to rotate view
                </div>
              </div>
            </div>
          </div>

          {/* Tracking Summary & Object List */}
          <div className="space-y-4">
            
            {/* Quick Stats */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Tracked Objects:</span>
                <span className="text-white font-medium">{trackingData.totalObjects}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Active Tracking:</span>
                <span className="font-medium" style={{ color: HEX_COLORS.status.good }}>
                  {trackingData.activeTracking}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Threats:</span>
                <span className="font-medium" style={{ color: HEX_COLORS.alerts.critical }}>
                  {trackingData.threats}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Primary Asset:</span>
                <span className="font-medium text-xs" style={{ color: HEX_COLORS.status.info }}>
                  {trackingData.primaryAsset}
                </span>
              </div>
            </div>

            <hr style={{ borderColor: HEX_COLORS.border }} />

            {/* Object List */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-white">Critical Objects</h4>
              {satellites
                .filter(sat => sat.threat_level !== 'none')
                .sort((a, b) => {
                  const threatOrder = { critical: 4, high: 3, medium: 2, low: 1, none: 0 }
                  return threatOrder[b.threat_level] - threatOrder[a.threat_level]
                })
                .map((satellite) => (
                  <div key={satellite.id} className="flex items-center justify-between p-2 border rounded"
                       style={{ 
                         backgroundColor: HEX_COLORS.background.secondary,
                         borderColor: HEX_COLORS.border
                       }}>
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: getObjectColor(satellite) }}
                      />
                      <div>
                        <div className="text-xs font-medium text-white">{satellite.id}</div>
                        <div className="text-xs text-gray-400">{satellite.altitude}km</div>
                      </div>
                    </div>
                    <Badge variant={getThreatVariant(satellite.threat_level)} className="text-xs">
                      {satellite.threat_level.toUpperCase()}
                    </Badge>
                  </div>
                ))}
            </div>

            {/* Control Actions */}
            <div className="space-y-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={handleRefreshTracking}
              >
                <RefreshCw className="h-3 w-3 mr-2" />
                Refresh Tracking
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={handleFocusPrimary}
              >
                <Target className="h-3 w-3 mr-2" />
                Focus Primary
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full"
                onClick={handleResetView}
              >
                <RotateCcw className="h-3 w-3 mr-2" />
                Reset View
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 