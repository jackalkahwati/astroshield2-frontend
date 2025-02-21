import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { styled } from '@mui/material/styles';
import { 
    Box, 
    Paper, 
    Typography, 
    Grid,
    Slider,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Switch,
    FormControlLabel
} from '@mui/material';

// Replace with your Mapbox token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || '';

const MapContainer = styled(Box)(({ theme }) => ({
    height: '600px',
    width: '100%',
    position: 'relative',
    borderRadius: theme.shape.borderRadius,
    overflow: 'hidden'
}));

const ControlPanel = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    marginBottom: theme.spacing(2)
}));

const DataPanel = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    height: '100%'
}));

interface TrajectoryPoint {
    time: number;
    position: [number, number, number];
    velocity: [number, number, number];
}

interface ImpactPrediction {
    time: string;
    location: {
        lat: number;
        lon: number;
    };
    velocity: {
        magnitude: number;
        direction: {
            x: number;
            y: number;
            z: number;
        };
    };
    uncertainty_radius_km: number;
    confidence: number;
    monte_carlo_stats?: {
        samples: number;
        time_std: number;
        position_std: number;
        velocity_std: number;
    };
}

interface TrajectoryViewerProps {
    trajectory: TrajectoryPoint[];
    impactPrediction: ImpactPrediction;
    breakupPoints?: {
        altitude: number;
        fragments: number;
        time: string;
    }[];
}

const TrajectoryViewer: React.FC<TrajectoryViewerProps> = ({
    trajectory,
    impactPrediction,
    breakupPoints
}) => {
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const [timeIndex, setTimeIndex] = useState(0);
    const [showUncertainty, setShowUncertainty] = useState(true);
    const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d');
    
    useEffect(() => {
        if (!mapContainer.current) return;
        
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/mapbox/satellite-v9',
            center: [impactPrediction.location.lon, impactPrediction.location.lat],
            zoom: 5,
            pitch: viewMode === '3d' ? 60 : 0
        });
        
        map.current.on('load', () => {
            // Add trajectory line
            map.current?.addSource('trajectory', {
                type: 'geojson',
                data: {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'LineString',
                        coordinates: trajectory.map(point => [
                            point.position[0],
                            point.position[1],
                            point.position[2]
                        ])
                    }
                }
            });
            
            map.current?.addLayer({
                id: 'trajectory-line',
                type: 'line',
                source: 'trajectory',
                layout: {
                    'line-join': 'round',
                    'line-cap': 'round'
                },
                paint: {
                    'line-color': '#ff4444',
                    'line-width': 2,
                    'line-opacity': 0.8
                }
            });
            
            // Add impact point
            map.current?.addSource('impact', {
                type: 'geojson',
                data: {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'Point',
                        coordinates: [
                            impactPrediction.location.lon,
                            impactPrediction.location.lat
                        ]
                    }
                }
            });
            
            map.current?.addLayer({
                id: 'impact-point',
                type: 'circle',
                source: 'impact',
                paint: {
                    'circle-radius': 8,
                    'circle-color': '#ff0000',
                    'circle-opacity': 0.8
                }
            });
            
            // Add uncertainty circle
            if (showUncertainty) {
                const uncertaintyCircle = createUncertaintyCircle(
                    [impactPrediction.location.lon, impactPrediction.location.lat],
                    impactPrediction.uncertainty_radius_km
                );
                
                map.current?.addSource('uncertainty', {
                    type: 'geojson',
                    data: uncertaintyCircle
                });
                
                map.current?.addLayer({
                    id: 'uncertainty-area',
                    type: 'fill',
                    source: 'uncertainty',
                    paint: {
                        'fill-color': '#ff0000',
                        'fill-opacity': 0.2
                    }
                });
            }
            
            // Add breakup points if available
            if (breakupPoints) {
                map.current?.addSource('breakups', {
                    type: 'geojson',
                    data: {
                        type: 'FeatureCollection',
                        features: breakupPoints.map((point, index) => ({
                            type: 'Feature',
                            properties: {
                                altitude: point.altitude,
                                fragments: point.fragments,
                                time: point.time
                            },
                            geometry: {
                                type: 'Point',
                                coordinates: [
                                    trajectory[index].position[0],
                                    trajectory[index].position[1]
                                ]
                            }
                        }))
                    }
                });
                
                map.current?.addLayer({
                    id: 'breakup-points',
                    type: 'symbol',
                    source: 'breakups',
                    layout: {
                        'icon-image': 'warning',
                        'icon-size': 1.5,
                        'text-field': ['get', 'fragments'],
                        'text-offset': [0, 1.5]
                    }
                });
            }
        });
        
        return () => {
            map.current?.remove();
        };
    }, [viewMode, showUncertainty]);
    
    const createUncertaintyCircle = (center: [number, number], radiusKm: number) => {
        const points = 64;
        const coords: [number, number][] = [];
        
        for (let i = 0; i < points; i++) {
            const angle = (i * 360) / points;
            const lat = center[1] + (radiusKm / 111.32) * Math.cos(angle * Math.PI / 180);
            const lon = center[0] + (radiusKm / (111.32 * Math.cos(center[1] * Math.PI / 180))) * Math.sin(angle * Math.PI / 180);
            coords.push([lon, lat]);
        }
        
        coords.push(coords[0]); // Close the circle
        
        return {
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'Polygon',
                coordinates: [coords]
            }
        };
    };
    
    const updateTrajectoryPosition = (index: number) => {
        if (!map.current) return;
        
        const point = trajectory[index];
        map.current.setCenter([point.position[0], point.position[1]]);
        
        // Update current position marker
        const source = map.current.getSource('current-position') as mapboxgl.GeoJSONSource;
        if (source) {
            source.setData({
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'Point',
                    coordinates: [point.position[0], point.position[1], point.position[2]]
                }
            });
        }
    };
    
    return (
        <Grid container spacing={2}>
            <Grid item xs={12}>
                <ControlPanel>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={6}>
                            <FormControl fullWidth>
                                <InputLabel>View Mode</InputLabel>
                                <Select
                                    value={viewMode}
                                    onChange={(e) => setViewMode(e.target.value as '2d' | '3d')}
                                >
                                    <MenuItem value="2d">2D View</MenuItem>
                                    <MenuItem value="3d">3D View</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={6}>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={showUncertainty}
                                        onChange={(e) => setShowUncertainty(e.target.checked)}
                                    />
                                }
                                label="Show Uncertainty"
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography gutterBottom>
                                Trajectory Timeline
                            </Typography>
                            <Slider
                                value={timeIndex}
                                onChange={(_, value) => {
                                    setTimeIndex(value as number);
                                    updateTrajectoryPosition(value as number);
                                }}
                                min={0}
                                max={trajectory.length - 1}
                                step={1}
                            />
                        </Grid>
                    </Grid>
                </ControlPanel>
            </Grid>
            
            <Grid item xs={9}>
                <MapContainer ref={mapContainer} />
            </Grid>
            
            <Grid item xs={3}>
                <DataPanel>
                    <Typography variant="h6" gutterBottom>
                        Impact Prediction
                    </Typography>
                    
                    <Typography variant="body2">
                        Time: {impactPrediction.time}
                    </Typography>
                    
                    <Typography variant="body2">
                        Location: {impactPrediction.location.lat.toFixed(4)}°N, {impactPrediction.location.lon.toFixed(4)}°E
                    </Typography>
                    
                    <Typography variant="body2">
                        Velocity: {impactPrediction.velocity.magnitude.toFixed(2)} m/s
                    </Typography>
                    
                    <Typography variant="body2">
                        Uncertainty: ±{impactPrediction.uncertainty_radius_km.toFixed(1)} km
                    </Typography>
                    
                    <Typography variant="body2">
                        Confidence: {(impactPrediction.confidence * 100).toFixed(1)}%
                    </Typography>
                    
                    {impactPrediction.monte_carlo_stats && (
                        <>
                            <Typography variant="subtitle2" sx={{ mt: 2 }}>
                                Monte Carlo Statistics
                            </Typography>
                            
                            <Typography variant="body2">
                                Samples: {impactPrediction.monte_carlo_stats.samples}
                            </Typography>
                            
                            <Typography variant="body2">
                                Time Std: ±{impactPrediction.monte_carlo_stats.time_std.toFixed(1)} s
                            </Typography>
                            
                            <Typography variant="body2">
                                Position Std: ±{impactPrediction.monte_carlo_stats.position_std.toFixed(1)} m
                            </Typography>
                            
                            <Typography variant="body2">
                                Velocity Std: ±{impactPrediction.monte_carlo_stats.velocity_std.toFixed(1)} m/s
                            </Typography>
                        </>
                    )}
                    
                    {breakupPoints && breakupPoints.length > 0 && (
                        <>
                            <Typography variant="subtitle2" sx={{ mt: 2 }}>
                                Breakup Events
                            </Typography>
                            
                            {breakupPoints.map((point, index) => (
                                <Box key={index} sx={{ mt: 1 }}>
                                    <Typography variant="body2">
                                        Time: {point.time}
                                    </Typography>
                                    <Typography variant="body2">
                                        Altitude: {point.altitude.toFixed(1)} km
                                    </Typography>
                                    <Typography variant="body2">
                                        Fragments: {point.fragments}
                                    </Typography>
                                </Box>
                            ))}
                        </>
                    )}
                </DataPanel>
            </Grid>
        </Grid>
    );
};

export default TrajectoryViewer; 