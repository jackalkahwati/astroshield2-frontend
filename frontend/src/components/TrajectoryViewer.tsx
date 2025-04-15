import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Play, Pause, FastForward, RotateCw, AlertTriangle } from 'lucide-react';

// Replace with your actual Mapbox token
mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || 'pk.eyJ1IjoiYXN0cm9zaGllbGQiLCJhIjoiY2xybDJtZGdvMWRtZjJqcnhkeXljbDUwNCJ9.QGrYKfnAWNVBZCxcRxp3_g';

interface TrajectoryPoint {
  time: number;
  position: [number, number, number]; // [longitude, latitude, altitude]
  velocity: [number, number, number]; // [vx, vy, vz]
  atmosphere?: {
    density: number;
    temperature: number;
  };
  metadata?: {
    status: string;
    warnings: string[];
  };
}

interface BreakupPoint {
  time: number;
  position: [number, number, number];
  fragments: number;
  cause: string;
}

interface ImpactPrediction {
  time: number;
  position: [number, number, number];
  confidence: number;
  energy: number;
  area: number;
  casualty_expectation?: number;
}

interface TrajectoryViewerProps {
  trajectory: TrajectoryPoint[];
  impactPrediction?: ImpactPrediction;
  breakupPoints?: BreakupPoint[];
  highlightedTimeRange?: [number, number];
  width?: string | number;
  height?: string | number;
  initialZoom?: number;
  showControls?: boolean;
  autoPlay?: boolean;
  playbackSpeed?: number;
}

const TrajectoryViewer: React.FC<TrajectoryViewerProps> = ({
  trajectory,
  impactPrediction,
  breakupPoints = [],
  highlightedTimeRange,
  width = '100%',
  height = '500px',
  initialZoom = 2,
  showControls = true,
  autoPlay = false,
  playbackSpeed = 1,
}) => {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [playing, setPlaying] = useState<boolean>(autoPlay);
  const [timeIndex, setTimeIndex] = useState<number>(0);
  const [speed, setSpeed] = useState<number>(playbackSpeed);
  const animationRef = useRef<number | null>(null);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: [trajectory[0].position[0], trajectory[0].position[1]],
      zoom: initialZoom,
      pitch: 40,
      bearing: 0,
      projection: { name: 'globe' }
    });

    map.current.on('load', () => {
      const mapInstance = map.current;
      if (!mapInstance) return;

      // Add trajectory line
      mapInstance.addSource('trajectory', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: trajectory.map(point => [
              point.position[0],
              point.position[1],
              point.position[2] / 1000 // Convert to km for better visualization
            ])
          }
        }
      });

      mapInstance.addLayer({
        id: 'trajectory-line',
        type: 'line',
        source: 'trajectory',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': '#4B91F7',
          'line-width': 3,
          'line-opacity': 0.8
        }
      });

      // Add impact prediction point if available
      if (impactPrediction) {
        mapInstance.addSource('impact-point', {
          type: 'geojson',
          data: {
            type: 'Feature',
            properties: {
              description: `Impact at ${new Date(impactPrediction.time * 1000).toLocaleString()}
                Energy: ${impactPrediction.energy.toFixed(2)} kJ
                Area: ${impactPrediction.area.toFixed(2)} km²
                Confidence: ${(impactPrediction.confidence * 100).toFixed(0)}%`
            },
            geometry: {
              type: 'Point',
              coordinates: [
                impactPrediction.position[0],
                impactPrediction.position[1],
                impactPrediction.position[2] / 1000
              ]
            }
          }
        });

        mapInstance.addLayer({
          id: 'impact-point',
          type: 'circle',
          source: 'impact-point',
          paint: {
            'circle-radius': 10,
            'circle-color': '#FF0000',
            'circle-opacity': 0.7,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#FFFFFF'
          }
        });

        // Add impact uncertainty circle
        mapInstance.addSource('impact-uncertainty', {
          type: 'geojson',
          data: {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Point',
              coordinates: [
                impactPrediction.position[0],
                impactPrediction.position[1]
              ]
            }
          }
        });

        mapInstance.addLayer({
          id: 'impact-uncertainty',
          type: 'circle',
          source: 'impact-uncertainty',
          paint: {
            'circle-radius': [
              'interpolate',
              ['linear'],
              ['zoom'],
              0, 50 * (1 - impactPrediction.confidence),
              12, 2 * (1 - impactPrediction.confidence) * 100000
            ],
            'circle-color': '#FF0000',
            'circle-opacity': 0.2,
            'circle-stroke-width': 1,
            'circle-stroke-color': '#FF0000',
            'circle-stroke-opacity': 0.5
          }
        });
      }

      // Add breakup points if available
      if (breakupPoints.length > 0) {
        const features = breakupPoints.map(point => ({
          type: 'Feature',
          properties: {
            description: `Breakup at ${new Date(point.time * 1000).toLocaleString()}
              Fragments: ${point.fragments}
              Cause: ${point.cause}`
          },
          geometry: {
            type: 'Point',
            coordinates: [
              point.position[0],
              point.position[1],
              point.position[2] / 1000
            ]
          }
        }));

        mapInstance.addSource('breakup-points', {
          type: 'geojson',
          data: {
            type: 'FeatureCollection',
            features
          }
        });

        mapInstance.addLayer({
          id: 'breakup-points',
          type: 'circle',
          source: 'breakup-points',
          paint: {
            'circle-radius': 8,
            'circle-color': '#FFA500',
            'circle-opacity': 0.9,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#FFFFFF'
          }
        });
      }

      // Add aircraft marker for current position
      mapInstance.addSource('aircraft', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'Point',
            coordinates: [
              trajectory[0].position[0],
              trajectory[0].position[1],
              trajectory[0].position[2] / 1000
            ]
          }
        }
      });

      mapInstance.addLayer({
        id: 'aircraft',
        type: 'symbol',
        source: 'aircraft',
        layout: {
          'icon-image': 'rocket',
          'icon-size': 1,
          'icon-rotate': ['get', 'rotation'],
          'icon-allow-overlap': true,
          'icon-ignore-placement': true
        }
      });

      // Add popup for aircraft
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      mapInstance.on('mouseenter', 'aircraft', () => {
        mapInstance.getCanvas().style.cursor = 'pointer';
        
        const coordinates = [
          trajectory[timeIndex].position[0],
          trajectory[timeIndex].position[1]
        ];
        
        const altitude = (trajectory[timeIndex].position[2] / 1000).toFixed(2);
        const speed = Math.sqrt(
          Math.pow(trajectory[timeIndex].velocity[0], 2) +
          Math.pow(trajectory[timeIndex].velocity[1], 2) +
          Math.pow(trajectory[timeIndex].velocity[2], 2)
        ).toFixed(2);
        
        const description = `
          <div>
            <strong>Time:</strong> ${new Date(trajectory[timeIndex].time * 1000).toLocaleString()}<br>
            <strong>Altitude:</strong> ${altitude} km<br>
            <strong>Speed:</strong> ${speed} m/s<br>
            <strong>Position:</strong> ${coordinates[0].toFixed(5)}, ${coordinates[1].toFixed(5)}
          </div>
        `;
        
        popup.setLngLat(coordinates as [number, number])
          .setHTML(description)
          .addTo(mapInstance);
      });
      
      mapInstance.on('mouseleave', 'aircraft', () => {
        mapInstance.getCanvas().style.cursor = '';
        popup.remove();
      });

      // Load satellite imagery
      mapInstance.addLayer({
        id: 'satellite',
        type: 'raster',
        source: {
          type: 'raster',
          url: 'mapbox://mapbox.satellite',
          tileSize: 256
        },
        minzoom: 0,
        maxzoom: 22
      }, 'trajectory-line');
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [trajectory, impactPrediction, breakupPoints, initialZoom]);

  // Update trajectory position
  const updateTrajectoryPosition = (index: number) => {
    if (!map.current || index >= trajectory.length) return;
    
    const point = trajectory[index];
    
    // Update aircraft position
    const aircraftSource = map.current.getSource('aircraft') as mapboxgl.GeoJSONSource;
    if (aircraftSource) {
      // Calculate rotation angle based on velocity
      const rotation = Math.atan2(point.velocity[1], point.velocity[0]) * (180 / Math.PI);
      
      aircraftSource.setData({
        type: 'Feature',
        properties: {
          rotation
        },
        geometry: {
          type: 'Point',
          coordinates: [
            point.position[0],
            point.position[1],
            point.position[2] / 1000
          ]
        }
      });
    }

    // Center map on current position
    map.current.easeTo({
      center: [point.position[0], point.position[1]],
      duration: 500
    });
  };

  // Animation loop
  useEffect(() => {
    if (playing) {
      let lastTime = 0;
      let accumulatedTime = 0;
      
      const animate = (time: number) => {
        if (!lastTime) lastTime = time;
        const deltaTime = time - lastTime;
        lastTime = time;
        
        accumulatedTime += deltaTime * speed;
        
        if (accumulatedTime > 200) { // Update every 200ms with adjustments for speed
          accumulatedTime = 0;
          setTimeIndex(prev => {
            if (prev >= trajectory.length - 1) {
              setPlaying(false);
              return prev;
            }
            return prev + 1;
          });
        }
        
        animationRef.current = requestAnimationFrame(animate);
      };
      
      animationRef.current = requestAnimationFrame(animate);
      
      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }
  }, [playing, trajectory.length, speed]);

  // Update position when timeIndex changes
  useEffect(() => {
    updateTrajectoryPosition(timeIndex);
  }, [timeIndex]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium">Trajectory Timeline</h3>
          {trajectory[timeIndex]?.metadata?.warnings?.length > 0 && (
            <Badge variant="destructive" className="flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              Warning
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div ref={mapContainer} style={{ width, height }} className="rounded-md overflow-hidden mb-4" />
        
        {showControls && (
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <button 
                onClick={() => setPlaying(!playing)}
                className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                {playing ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
              </button>
              
              <button 
                onClick={() => setSpeed(prev => Math.min(prev * 2, 8))}
                className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
                disabled={speed >= 8}
              >
                <FastForward className="h-5 w-5" />
              </button>
              
              <button 
                onClick={() => setTimeIndex(0)}
                className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                <RotateCw className="h-5 w-5" />
              </button>
              
              <div className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                {speed}x Speed | {new Date(trajectory[timeIndex]?.time * 1000).toLocaleString()}
              </div>
            </div>
            
            <Slider
              value={[timeIndex]}
              min={0}
              max={trajectory.length - 1}
              step={1}
              onValueChange={(value) => {
                setTimeIndex(value[0]);
                setPlaying(false);
              }}
            />
            
            {trajectory[timeIndex]?.metadata?.warnings?.length > 0 && (
              <div className="mt-2 text-red-500 text-sm">
                {trajectory[timeIndex].metadata.warnings.map((warning, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" /> {warning}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TrajectoryViewer; 