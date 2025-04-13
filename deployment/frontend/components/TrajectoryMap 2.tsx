"use client"

import React, { useRef, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { Button } from "@/components/ui/button"
import { Plus, Minus, Navigation, Layers, MapPin, Eye, EyeOff } from 'lucide-react'

// Use environment variable for Mapbox token
mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || 'pk.eyJ1IjoiaXExOXplcm8xMiIsImEiOiJjajNveDZkNWMwMGtpMnFuNG05MjNidjBrIn0.rbEk-JO7ewQXACGoTCT5CQ'

import { TrajectoryPoint, BreakupEvent, ImpactPrediction } from "@/lib/types";

interface TrajectoryMapProps {
  showUncertainty?: boolean
  darkMode?: boolean
  timeIndex?: number
  maxTimeIndex?: number
  trajectoryData?: {
    trajectory: TrajectoryPoint[];
    impact_prediction: ImpactPrediction;
    breakup_events?: BreakupEvent[];
  }
  isLoading?: boolean
}

// Mock trajectory data for demonstration (used when real data is not available)
const MOCK_TRAJECTORY = Array(100).fill(0).map((_, i) => ({
  time: i * 10,
  position: [
    -74.0060 + (i * 0.01),
    40.7128 + (i * 0.01),
    100000 - (i * 1000)
  ] as [number, number, number],
  velocity: [
    1000 - (i * 10),
    0,
    -1000 - (i * 10)
  ] as [number, number, number]
}))

const MOCK_IMPACT = {
  time: new Date().toISOString(),
  location: {
    lat: 41.7128,
    lon: -73.0060
  },
  velocity: {
    magnitude: 1000,
    direction: {
      x: 0.7,
      y: 0,
      z: -0.7
    }
  },
  uncertainty_radius_km: 10,
  confidence: 0.95
}

const MOCK_BREAKUP_POINTS = [
  {
    altitude: 80000,
    fragments: 5,
    time: new Date().toISOString()
  },
  {
    altitude: 60000,
    fragments: 10,
    time: new Date().toISOString()
  }
]

const TrajectoryMap: React.FC<TrajectoryMapProps> = ({
  showUncertainty = true,
  darkMode = true,
  timeIndex = 0,
  maxTimeIndex = 100,
  trajectoryData,
  isLoading = false
}) => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d')
  
  // Use real data if available, otherwise use mock data
  const trajectory = trajectoryData?.trajectory || MOCK_TRAJECTORY
  const impactPrediction = trajectoryData?.impact_prediction || MOCK_IMPACT
  const breakupEvents = trajectoryData?.breakup_events || MOCK_BREAKUP_POINTS
  
  const [currentPosition, setCurrentPosition] = useState<[number, number]>([
    trajectory[0].position[0],
    trajectory[0].position[1]
  ])
  
  // Calculate the actual trajectory index based on maxTimeIndex
  const actualIndex = Math.min(
    Math.floor((timeIndex / maxTimeIndex) * trajectory.length),
    trajectory.length - 1
  )

  // Generate a circle polygon for uncertainty visualization
  const createUncertaintyCircle = (center: [number, number], radiusKm: number) => {
    const points = 64
    const coords: [number, number][] = []
    
    for (let i = 0; i < points; i++) {
      const angle = (i * 360) / points
      const lat = center[1] + (radiusKm / 111.32) * Math.cos(angle * Math.PI / 180)
      const lon = center[0] + (radiusKm / (111.32 * Math.cos(center[1] * Math.PI / 180))) * Math.sin(angle * Math.PI / 180)
      coords.push([lon, lat])
    }
    
    coords.push(coords[0]) // Close the circle
    
    return {
      type: 'Feature' as const,
      properties: {},
      geometry: {
        type: 'Polygon' as const,
        coordinates: [coords]
      }
    }
  }

  // Initialize map when component mounts
  useEffect(() => {
    if (!mapContainer.current) return

    // Choose map style based on dark mode
    const mapStyle = darkMode ? 'mapbox://styles/mapbox/dark-v11' : 'mapbox://styles/mapbox/light-v11'
    
    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: mapStyle,
        center: [MOCK_TRAJECTORY[0].position[0], MOCK_TRAJECTORY[0].position[1]],
        zoom: 5,
        pitch: viewMode === '3d' ? 60 : 0
      })

      // Handle map load errors
      map.current.on('error', (err) => {
        console.error('Map error:', err);
      });

      // Add navigation controls
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-left')
      
      // Wait for map to load
      map.current.on('load', () => {
        if (!map.current) return
        
        try {
          // Add trajectory line
          map.current.addSource('trajectory', {
            type: 'geojson',
            data: {
              type: 'Feature',
              properties: {},
              geometry: {
                type: 'LineString',
                coordinates: MOCK_TRAJECTORY.map(point => [
                  point.position[0],
                  point.position[1]
                ])
              }
            }
          })
          
          map.current.addLayer({
            id: 'trajectory-line',
            type: 'line',
            source: 'trajectory',
            layout: {
              'line-join': 'round',
              'line-cap': 'round'
            },
            paint: {
              'line-color': darkMode ? '#3b82f6' : '#2563eb', // blue-500/600
              'line-width': 3,
              'line-opacity': 0.8
            }
          })
          
          // Add impact point
          map.current.addSource('impact', {
            type: 'geojson',
            data: {
              type: 'Feature',
              properties: {},
              geometry: {
                type: 'Point',
                coordinates: [
                  MOCK_IMPACT.location.lon,
                  MOCK_IMPACT.location.lat
                ]
              }
            }
          })
          
          map.current.addLayer({
            id: 'impact-point',
            type: 'circle',
            source: 'impact',
            paint: {
              'circle-radius': 10,
              'circle-color': '#ef4444', // red-500
              'circle-opacity': 0.8,
              'circle-stroke-width': 2,
              'circle-stroke-color': '#ffffff'
            }
          })

          // Add uncertainty circle
          const uncertaintyCircle = createUncertaintyCircle(
            [MOCK_IMPACT.location.lon, MOCK_IMPACT.location.lat],
            MOCK_IMPACT.uncertainty_radius_km
          )
          
          map.current.addSource('uncertainty', {
            type: 'geojson',
            data: uncertaintyCircle
          })
          
          map.current.addLayer({
            id: 'uncertainty-area',
            type: 'fill',
            source: 'uncertainty',
            paint: {
              'fill-color': '#ef4444', // red-500
              'fill-opacity': 0.2,
              'fill-outline-color': '#ef4444'
            },
            layout: {
              visibility: showUncertainty ? 'visible' : 'none'
            }
          })
          
          // Add uncertainty border
          map.current.addLayer({
            id: 'uncertainty-border',
            type: 'line',
            source: 'uncertainty',
            paint: {
              'line-color': '#ef4444', // red-500
              'line-width': 2,
              'line-dasharray': [3, 3]
            },
            layout: {
              visibility: showUncertainty ? 'visible' : 'none'
            }
          })
          
          // Add current position marker
          map.current.addSource('current-position', {
            type: 'geojson',
            data: {
              type: 'Feature',
              properties: {},
              geometry: {
                type: 'Point',
                coordinates: [
                  MOCK_TRAJECTORY[0].position[0],
                  MOCK_TRAJECTORY[0].position[1]
                ]
              }
            }
          })
          
          map.current.addLayer({
            id: 'current-position-point',
            type: 'circle',
            source: 'current-position',
            paint: {
              'circle-radius': 8,
              'circle-color': '#3b82f6', // blue-500
              'circle-opacity': 1,
              'circle-stroke-width': 2,
              'circle-stroke-color': '#ffffff'
            }
          })
          
          // Add breakup points
          map.current.addSource('breakups', {
            type: 'geojson',
            data: {
              type: 'FeatureCollection',
              features: MOCK_BREAKUP_POINTS.map((point, index) => ({
                type: 'Feature',
                properties: {
                  fragments: point.fragments
                },
                geometry: {
                  type: 'Point',
                  coordinates: [
                    MOCK_TRAJECTORY[Math.min(index * 20, MOCK_TRAJECTORY.length - 1)].position[0],
                    MOCK_TRAJECTORY[Math.min(index * 20, MOCK_TRAJECTORY.length - 1)].position[1]
                  ]
                }
              }))
            }
          })
          
          map.current.addLayer({
            id: 'breakup-points',
            type: 'circle',
            source: 'breakups',
            paint: {
              'circle-radius': 10,
              'circle-color': '#f59e0b', // amber-500
              'circle-opacity': 0.9,
              'circle-stroke-width': 2,
              'circle-stroke-color': '#ffffff'
            }
          })
          
          // Add pulse animation for current position
          map.current.addLayer({
            id: 'current-position-pulse',
            type: 'circle',
            source: 'current-position',
            paint: {
              'circle-radius': [
                'interpolate',
                ['linear'],
                ['get', 'pulse'],
                0, 8,
                1, 20
              ],
              'circle-color': '#3b82f6', // blue-500
              'circle-opacity': [
                'interpolate',
                ['linear'],
                ['get', 'pulse'],
                0, 0.6,
                1, 0
              ],
              'circle-stroke-width': 0
            }
          })

          setMapLoaded(true)
        } catch (error) {
          console.error('Error initializing map layers:', error);
        }
      })
    } catch (error) {
      console.error('Error initializing map:', error);
    }

    return () => {
      try {
        map.current?.remove()
      } catch (error) {
        console.error('Error removing map:', error);
      }
    }
  }, [darkMode])

  // Update uncertainty visibility when showUncertainty changes
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    
    if (map.current.getLayer('uncertainty-area')) {
      map.current.setLayoutProperty(
        'uncertainty-area',
        'visibility',
        showUncertainty ? 'visible' : 'none'
      )
    }
    
    if (map.current.getLayer('uncertainty-border')) {
      map.current.setLayoutProperty(
        'uncertainty-border',
        'visibility',
        showUncertainty ? 'visible' : 'none'
      )
    }
  }, [showUncertainty, mapLoaded])

  // Update view mode (2D/3D)
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    
    map.current.easeTo({
      pitch: viewMode === '3d' ? 60 : 0,
      duration: 1000
    })
  }, [viewMode, mapLoaded])
  
  // Update position when timeIndex changes
  useEffect(() => {
    if (!map.current || !mapLoaded) return
    updateCurrentPosition(actualIndex)
  }, [actualIndex, mapLoaded])

  // Function to update the current position marker
  const updateCurrentPosition = (index: number) => {
    if (!map.current || !mapLoaded) return
    
    const point = MOCK_TRAJECTORY[index]
    setCurrentPosition([point.position[0], point.position[1]])
    
    // Update current position marker
    const source = map.current.getSource('current-position') as mapboxgl.GeoJSONSource
    if (source) {
      source.setData({
        type: 'Feature',
        properties: {
          pulse: (Date.now() % 1000) / 1000 // For pulse animation
        },
        geometry: {
          type: 'Point',
          coordinates: [point.position[0], point.position[1]]
        }
      })
    }
    
    // Pan map to follow the current position
    map.current.panTo([point.position[0], point.position[1]], { duration: 500 })
  }

  // Handler for timeline slider changes - we don't handle timeIndex updates here
  // as timeIndex is a prop passed to this component
  const handleTimeIndexChange = (newIndex: number) => {
    // We don't have a setTimeIndex function, as timeIndex is a prop
    updateCurrentPosition(newIndex)
  }

  // Function to zoom in
  const zoomIn = () => {
    if (!map.current) return
    map.current.zoomIn()
  }

  // Function to zoom out
  const zoomOut = () => {
    if (!map.current) return
    map.current.zoomOut()
  }

  // Function to toggle 2D/3D view
  const toggleViewMode = () => {
    setViewMode(prev => prev === '2d' ? '3d' : '2d')
  }

  // Function to focus on impact point
  const focusOnImpact = () => {
    if (!map.current) return
    map.current.flyTo({
      center: [MOCK_IMPACT.location.lon, MOCK_IMPACT.location.lat],
      zoom: 7,
      duration: 1500
    })
  }

  return (
    <div className="relative h-full w-full">
      <div ref={mapContainer} className="h-full w-full" />
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-white/50 dark:bg-slate-900/50 flex items-center justify-center backdrop-blur-sm z-50">
          <div className="flex flex-col items-center gap-3 p-4 rounded-lg bg-white dark:bg-slate-800 shadow-lg">
            <div className="h-8 w-8 rounded-full border-4 border-t-blue-600 border-blue-200 animate-spin"></div>
            <p className="text-sm font-medium text-slate-900 dark:text-white">Processing trajectory data...</p>
          </div>
        </div>
      )}
      
      {/* Floating controls */}
      <div className="absolute bottom-2 right-2 flex flex-col gap-2">
        <Button
          variant="secondary"
          size="sm"
          className="h-8 w-8 p-0 rounded-full bg-white/80 dark:bg-slate-800/80 text-slate-900 dark:text-white"
          onClick={zoomIn}
          disabled={isLoading}
        >
          <Plus className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="h-8 w-8 p-0 rounded-full bg-white/80 dark:bg-slate-800/80 text-slate-900 dark:text-white"
          onClick={zoomOut}
          disabled={isLoading}
        >
          <Minus className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="absolute top-2 right-2 flex flex-col gap-2">
        <Button
          variant="secondary"
          size="sm"
          className="h-8 w-8 p-0 rounded-full bg-white/80 dark:bg-slate-800/80 text-slate-900 dark:text-white"
          onClick={toggleViewMode}
          title={viewMode === '3d' ? 'Switch to 2D view' : 'Switch to 3D view'}
          disabled={isLoading}
        >
          <Layers className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="h-8 w-8 p-0 rounded-full bg-white/80 dark:bg-slate-800/80 text-slate-900 dark:text-white"
          onClick={focusOnImpact}
          title="Focus on impact point"
          disabled={isLoading}
        >
          <MapPin className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

export default TrajectoryMap