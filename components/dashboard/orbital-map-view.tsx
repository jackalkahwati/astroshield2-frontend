"use client"

import React, { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Globe, 
  Satellite, 
  AlertTriangle, 
  Target, 
  RefreshCw, 
  Radar,
  MapPin,
  Eye,
  Layers
} from 'lucide-react'

// We'll implement Mapbox functionality in useEffect to avoid SSR issues
interface SatelliteObject {
  id: string
  name: string
  latitude: number
  longitude: number
  altitude: number
  velocity: number
  type: 'active' | 'debris' | 'threat'
  threat_level: 'none' | 'low' | 'medium' | 'high' | 'critical'
  groundTrack?: Array<[number, number]>
}

interface GroundStation {
  id: string
  name: string
  latitude: number
  longitude: number
  type: 'dsn' | 'military' | 'commercial'
  active: boolean
}

interface TrackingData {
  totalObjects: number
  activeTracking: number
  threats: number
  primaryAsset: string
  coverage: string
}

export const OrbitalMapView: React.FC = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [mapboxLoaded, setMapboxLoaded] = useState(false)
  const [selectedSatellite, setSelectedSatellite] = useState<string | null>(null)
  const [showGroundTracks, setShowGroundTracks] = useState(true)
  const [showCoverage, setShowCoverage] = useState(false)

  const [trackingData, setTrackingData] = useState<TrackingData>({
    totalObjects: 47,
    activeTracking: 45,
    threats: 2,
    primaryAsset: "USA-317 (ISS)",
    coverage: "Global"
  })

  // Sample satellite data with realistic coordinates
  const [satellites, setSatellites] = useState<SatelliteObject[]>([
    {
      id: "USA-317",
      name: "International Space Station",
      latitude: 51.6461,
      longitude: -0.1272, // Over London initially
      altitude: 408,
      velocity: 7.66,
      type: 'active',
      threat_level: 'none',
      groundTrack: [
        [-0.1272, 51.6461],
        [2.3522, 48.8566], // Paris
        [13.4050, 52.5200], // Berlin
        [30.5234, 50.4501], // Kyiv
      ]
    },
    {
      id: "STARLINK-4729",
      name: "Starlink-4729",
      latitude: 37.7749,
      longitude: -122.4194, // Over San Francisco
      altitude: 547,
      velocity: 7.5,
      type: 'active',
      threat_level: 'medium',
      groundTrack: [
        [-122.4194, 37.7749],
        [-118.2437, 34.0522], // LA
        [-112.0740, 33.4484], // Phoenix
        [-104.9903, 39.7392], // Denver
      ]
    },
    {
      id: "DEB-001",
      name: "Unknown Debris",
      latitude: 35.6762,
      longitude: 139.6503, // Over Tokyo
      altitude: 375,
      velocity: 7.8,
      type: 'debris',
      threat_level: 'high',
      groundTrack: [
        [139.6503, 35.6762],
        [144.9631, -37.8136], // Melbourne
        [151.2093, -33.8688], // Sydney
        [174.7633, -36.8485], // Auckland
      ]
    },
    {
      id: "COSMOS-1408-DEB",
      name: "Cosmos 1408 Fragment", 
      latitude: 55.7558,
      longitude: 37.6176, // Over Moscow
      altitude: 485,
      velocity: 7.4,
      type: 'debris',
      threat_level: 'critical',
      groundTrack: [
        [37.6176, 55.7558],
        [49.8671, 40.4093], // Baku
        [69.2401, 41.2995], // Tashkent
        [76.9628, 43.2551], // Almaty
      ]
    }
  ])

  // Ground stations data
  const groundStations: GroundStation[] = [
    {
      id: "GDSS-1",
      name: "Goldstone Deep Space Communications Complex",
      latitude: 35.4264,
      longitude: -116.8905,
      type: 'dsn',
      active: true
    },
    {
      id: "CHDR-1", 
      name: "Cheyenne Mountain Space Force Station",
      latitude: 38.7444,
      longitude: -104.8460,
      type: 'military',
      active: true
    },
    {
      id: "VAFB-1",
      name: "Vandenberg Space Force Base",
      latitude: 34.7420,
      longitude: -120.5724,
      type: 'military', 
      active: true
    },
    {
      id: "ESOC-1",
      name: "European Space Operations Centre",
      latitude: 49.8728,
      longitude: 8.6512,
      type: 'commercial',
      active: true
    }
  ]

  useEffect(() => {
    // Dynamically import Mapbox to avoid SSR issues
    const initializeMap = async () => {
      try {
        // Check if we have a valid token
        const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN
        if (!token || token === 'pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw') {
          console.warn('No valid Mapbox token found, showing fallback view')
          setIsLoading(false)
          setMapboxLoaded(false)
          return
        }

        const mapboxgl = await import('mapbox-gl')
        
        // Set access token
        mapboxgl.default.accessToken = token

        if (map.current) return // Initialize map only once

        if (!mapContainer.current) {
          setIsLoading(false)
          return
        }

        map.current = new mapboxgl.default.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/satellite-v9', // Satellite imagery for space ops
          center: [0, 30], // Center on equator
          zoom: 2,
          projection: 'globe' as any, // Globe projection for better space context
        })

        map.current.on('load', () => {
          setMapboxLoaded(true)
          setIsLoading(false)
          
          // Add satellite markers
          satellites.forEach(satellite => {
            addSatelliteToMap(satellite)
          })

          // Add ground stations
          groundStations.forEach(station => {
            addGroundStationToMap(station)
          })

          // Add ground tracks if enabled
          if (showGroundTracks) {
            addGroundTracks()
          }
        })

        map.current.on('error', (e) => {
          console.error('Mapbox error:', e)
          setIsLoading(false)
          setMapboxLoaded(false)
        })

        // Add navigation controls
        map.current.addControl(new mapboxgl.default.NavigationControl())
        
      } catch (error) {
        console.error('Error loading Mapbox:', error)
        setIsLoading(false)
        setMapboxLoaded(false)
      }
    }

    // Set a timeout to prevent infinite loading
    const timeoutId = setTimeout(() => {
      if (isLoading) {
        console.warn('Mapbox loading timeout, showing fallback')
        setIsLoading(false)
        setMapboxLoaded(false)
      }
    }, 5000) // 5 second timeout

    initializeMap()

    // Cleanup
    return () => {
      clearTimeout(timeoutId)
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, [])

  const addSatelliteToMap = (satellite: SatelliteObject) => {
    if (!map.current || !mapboxLoaded) return

    const color = getObjectColor(satellite)
    const size = satellite.threat_level === 'critical' ? 12 : 
                satellite.threat_level === 'high' ? 10 : 8

    // Create satellite marker
    const el = document.createElement('div')
    el.className = 'satellite-marker'
    el.style.cssText = `
      background-color: ${color};
      width: ${size}px;
      height: ${size}px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 0 10px ${color};
      cursor: pointer;
      transition: transform 0.2s;
    `

    el.addEventListener('mouseenter', () => {
      el.style.transform = 'scale(1.5)'
    })

    el.addEventListener('mouseleave', () => {
      el.style.transform = 'scale(1)'
    })

    el.addEventListener('click', () => {
      setSelectedSatellite(satellite.id)
      // Fly to satellite location
      map.current.flyTo({
        center: [satellite.longitude, satellite.latitude],
        zoom: 8,
        duration: 2000
      })
    })

    // Add marker to map
    const mapboxgl = require('mapbox-gl')
    new mapboxgl.Marker(el)
      .setLngLat([satellite.longitude, satellite.latitude])
      .setPopup(
        new mapboxgl.Popup({ offset: 25 })
          .setHTML(`
            <div class="bg-gray-900 text-white p-3 rounded">
              <h3 class="font-bold text-sm">${satellite.name}</h3>
              <p class="text-xs">ID: ${satellite.id}</p>
              <p class="text-xs">Altitude: ${satellite.altitude}km</p>
              <p class="text-xs">Velocity: ${satellite.velocity}km/s</p>
              <p class="text-xs">Threat: <span class="font-bold">${satellite.threat_level.toUpperCase()}</span></p>
            </div>
          `)
      )
      .addTo(map.current)
  }

  const addGroundStationToMap = (station: GroundStation) => {
    if (!map.current || !mapboxLoaded) return

    const color = station.type === 'military' ? '#EF4444' : 
                 station.type === 'dsn' ? '#3B82F6' : '#10B981'

    const el = document.createElement('div')
    el.className = 'ground-station-marker'
    el.innerHTML = `
      <div style="
        background-color: ${color};
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 2px solid white;
        position: relative;
      ">
        <div style="
          position: absolute;
          top: -20px;
          left: -10px;
          width: 36px;
          height: 36px;
          border: 2px solid ${color};
          border-radius: 50%;
          opacity: 0.3;
          animation: pulse 2s infinite;
        "></div>
      </div>
    `

    const mapboxgl = require('mapbox-gl')
    new mapboxgl.Marker(el)
      .setLngLat([station.longitude, station.latitude])
      .setPopup(
        new mapboxgl.Popup({ offset: 25 })
          .setHTML(`
            <div class="bg-gray-900 text-white p-3 rounded">
              <h3 class="font-bold text-sm">${station.name}</h3>
              <p class="text-xs">Type: ${station.type.toUpperCase()}</p>
              <p class="text-xs">Status: ${station.active ? 'ACTIVE' : 'OFFLINE'}</p>
            </div>
          `)
      )
      .addTo(map.current)
  }

  const addGroundTracks = () => {
    if (!map.current || !mapboxLoaded) return

    satellites.forEach(satellite => {
      if (satellite.groundTrack) {
        map.current.addSource(`ground-track-${satellite.id}`, {
          type: 'geojson',
          data: {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'LineString',
              coordinates: satellite.groundTrack
            }
          }
        })

        map.current.addLayer({
          id: `ground-track-${satellite.id}`,
          type: 'line',
          source: `ground-track-${satellite.id}`,
          layout: {
            'line-join': 'round',
            'line-cap': 'round'
          },
          paint: {
            'line-color': getObjectColor(satellite),
            'line-width': 2,
            'line-opacity': 0.7
          }
        })
      }
    })
  }

  const getObjectColor = (satellite: SatelliteObject) => {
    if (satellite.threat_level === 'critical') return '#EF4444'
    if (satellite.threat_level === 'high') return '#F59E0B'
    if (satellite.threat_level === 'medium') return '#F59E0B'
    if (satellite.type === 'active') return '#10B981'
    return '#6B7280'
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
    // Simulate updating satellite positions
    setSatellites(prevSats => 
      prevSats.map(sat => ({
        ...sat,
        latitude: sat.latitude + (Math.random() - 0.5) * 2,
        longitude: sat.longitude + (Math.random() - 0.5) * 2,
      }))
    )
    
    setTrackingData(prev => ({
      ...prev,
      totalObjects: 45 + Math.floor(Math.random() * 5),
      activeTracking: 43 + Math.floor(Math.random() * 5),
      threats: Math.floor(Math.random() * 5)
    }))
  }

  const toggleGroundTracks = () => {
    setShowGroundTracks(!showGroundTracks)
    // Implementation would toggle ground track layers
  }

  const toggleCoverage = () => {
    setShowCoverage(!showCoverage)
    // Implementation would toggle coverage areas
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Orbital Situational Awareness</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-[#0A0E1A] border border-gray-700 rounded flex items-center justify-center">
            <div className="text-center space-y-4">
              <RefreshCw className="h-12 w-12 mx-auto text-blue-400 animate-spin" />
              <p className="text-white">Loading orbital tracking data...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Fallback view when Mapbox is not available
  if (!mapboxLoaded) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Orbital Situational Awareness</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Fallback View */}
            <div className="lg:col-span-2">
              <div className="h-96 bg-[#0A0E1A] border border-gray-700 rounded relative">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <Globe className="h-16 w-16 mx-auto text-blue-400" />
                    <div className="text-white">
                      <h3 className="text-lg font-semibold mb-2">Satellite Tracking Active</h3>
                      <p className="text-sm text-gray-400 mb-4">Real-time orbital monitoring</p>
                      <div className="grid grid-cols-2 gap-4 text-left max-w-md">
                        {satellites.slice(0, 4).map((satellite, index) => (
                          <div key={satellite.id} className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                            <div 
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: getObjectColor(satellite) }}
                            />
                            <div>
                              <div className="text-xs font-medium text-white">{satellite.id}</div>
                              <div className="text-xs text-gray-400">{satellite.altitude}km</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Status overlay */}
                <div className="absolute top-4 right-4 bg-[#1A1F2E] border border-gray-700 rounded p-3">
                  <div className="text-xs space-y-1">
                    <div className="flex items-center gap-2">
                      <Radar className="h-3 w-3 text-green-400" />
                      <span className="text-green-400">TRACKING ACTIVE</span>
                    </div>
                    <div className="text-white">Coverage: Global</div>
                    <div className="text-xs text-gray-400">Fallback Mode</div>
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
                  <span className="text-green-400 font-medium">{trackingData.activeTracking}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Threats:</span>
                  <span className="text-red-400 font-medium">{trackingData.threats}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">Primary Asset:</span>
                  <span className="text-blue-400 font-medium text-xs">{trackingData.primaryAsset}</span>
                </div>
              </div>

              <hr className="border-gray-700" />

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
                    <div 
                      key={satellite.id} 
                      className="flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded"
                    >
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
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-white">Orbital Situational Awareness</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Interactive Map View */}
          <div className="lg:col-span-2">
            <div className="relative">
              <div 
                ref={mapContainer} 
                className="h-96 rounded border border-gray-700"
              />
              
              {/* Map Controls Overlay */}
              <div className="absolute top-4 left-4 bg-[#1A1F2E] border border-gray-700 rounded p-3 space-y-2">
                <div className="text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <Radar className="h-3 w-3 text-green-400" />
                    <span className="text-green-400">TRACKING ACTIVE</span>
                  </div>
                  <div className="text-white">Coverage: Global</div>
                </div>
                
                <div className="space-y-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full text-xs h-8"
                    onClick={toggleGroundTracks}
                  >
                    <Layers className="h-3 w-3 mr-1" />
                    {showGroundTracks ? 'Hide' : 'Show'} Tracks
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full text-xs h-8"
                    onClick={toggleCoverage}
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    {showCoverage ? 'Hide' : 'Show'} Coverage
                  </Button>
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
                <span className="text-green-400 font-medium">{trackingData.activeTracking}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Threats:</span>
                <span className="text-red-400 font-medium">{trackingData.threats}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-white">Primary Asset:</span>
                <span className="text-blue-400 font-medium text-xs">{trackingData.primaryAsset}</span>
              </div>
            </div>

            <hr className="border-gray-700" />

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
                  <div 
                    key={satellite.id} 
                    className={`flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded cursor-pointer hover:border-gray-600 transition-colors ${
                      selectedSatellite === satellite.id ? 'border-blue-500' : ''
                    }`}
                    onClick={() => setSelectedSatellite(satellite.id)}
                  >
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

            {/* Ground Stations */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-white">Ground Stations</h4>
              {groundStations.slice(0, 3).map((station) => (
                <div key={station.id} className="flex items-center justify-between p-2 bg-[#1A1F2E] border border-gray-800 rounded">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-3 w-3 text-blue-400" />
                    <div>
                      <div className="text-xs font-medium text-white">{station.name.split(' ')[0]}</div>
                      <div className="text-xs text-gray-400">{station.type.toUpperCase()}</div>
                    </div>
                  </div>
                  <Badge variant={station.active ? "default" : "outline"} className="text-xs">
                    {station.active ? 'ACTIVE' : 'OFFLINE'}
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
                onClick={() => {
                  if (map.current) {
                    map.current.flyTo({
                      center: [0, 30],
                      zoom: 2,
                      duration: 2000
                    })
                  }
                }}
              >
                <Globe className="h-3 w-3 mr-2" />
                Global View
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 