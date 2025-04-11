import React, { useEffect, useRef, useState, useCallback } from 'react';
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
    FormControlLabel,
    Card,
    CardContent,
    Tabs,
    Tab,
    Tooltip,
    IconButton,
    ButtonGroup,
    Button,
    Chip,
    Divider,
    Stack,
    useTheme
} from '@mui/material';
import {
    PlayArrow,
    Pause,
    SkipNext,
    SkipPrevious,
    ViewInAr,
    Map as MapIcon,
    Warning,
    Speed,
    MyLocation,
    ZoomIn,
    ZoomOut,
    RestartAlt,
    VisibilityOff,
    Visibility
} from '@mui/icons-material';

// Use the environment variable for the Mapbox token
mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || process.env.MAPBOX_TOKEN || 'pk.eyJ1IjoiaXExOXplcm8xMiIsImEiOiJjajNveDZkNWMwMGtpMnFuNG05MjNidjBrIn0.rbEk-JO7ewQXACGoTCT5CQ';

const MapContainer = styled(Box)(({ theme }) => ({
    height: '600px',
    width: '100%',
    position: 'relative',
    borderRadius: theme.shape.borderRadius,
    overflow: 'hidden',
    boxShadow: theme.shadows[2]
}));

const ControlPanel = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    marginBottom: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[1]
}));

const DataPanel = styled(Card)(({ theme }) => ({
    height: '100%',
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[2]
}));

const DataItem = styled(Box)(({ theme }) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: theme.spacing(0.5, 0),
    '&:not(:last-child)': {
        borderBottom: `1px solid ${theme.palette.divider}`
    }
}));

const MapControls = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: theme.spacing(2),
    right: theme.spacing(2),
    zIndex: 1,
    background: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    padding: theme.spacing(0.5),
    boxShadow: theme.shadows[2]
}));

const PlaybackControls = styled(Box)(({ theme }) => ({
    position: 'absolute',
    bottom: theme.spacing(2),
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 1,
    background: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    padding: theme.spacing(1),
    boxShadow: theme.shadows[3],
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1)
}));

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`trajectory-view-tabpanel-${index}`}
            aria-labelledby={`trajectory-view-tab-${index}`}
            {...other}
            style={{ padding: '8px 0' }}
        >
            {value === index && (
                <Box>{children}</Box>
            )}
        </div>
    );
}

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

const mapStyles = [
    { id: 'satellite-v9', name: 'Satellite' },
    { id: 'light-v10', name: 'Light' },
    { id: 'dark-v10', name: 'Dark' },
    { id: 'outdoors-v11', name: 'Outdoors' },
    { id: 'navigation-day-v1', name: 'Navigation' }
];

const TrajectoryViewer: React.FC<TrajectoryViewerProps> = ({
    trajectory,
    impactPrediction,
    breakupPoints
}) => {
    const theme = useTheme();
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const [timeIndex, setTimeIndex] = useState(0);
    const [showUncertainty, setShowUncertainty] = useState(true);
    const [showBreakups, setShowBreakups] = useState(true);
    const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d');
    const [tabValue, setTabValue] = useState(0);
    const [mapStyle, setMapStyle] = useState('satellite-v9');
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [currentPoint, setCurrentPoint] = useState<TrajectoryPoint | null>(null);
    const animationRef = useRef<number | null>(null);

    const createMapControls = useCallback(() => {
        if (!map.current) return;

        // Add navigation control
        const nav = new mapboxgl.NavigationControl({
            visualizePitch: true
        });
        map.current.addControl(nav, 'top-left');

        // Add scale control
        const scale = new mapboxgl.ScaleControl({
            maxWidth: 100,
            unit: 'metric'
        });
        map.current.addControl(scale, 'bottom-left');

        // Add full screen control
        const fullscreen = new mapboxgl.FullscreenControl();
        map.current.addControl(fullscreen, 'top-left');
    }, []);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const formatDateTimeString = (dateString: string) => {
        try {
            const date = new Date(dateString);
            return date.toLocaleString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        } catch (e) {
            return dateString;
        }
    };

    const calculateAltitude = (position: [number, number, number]): number => {
        const earthRadius = 6371000; // in meters
        return Math.max(0, Math.sqrt(position[0]**2 + position[1]**2 + position[2]**2) - earthRadius);
    };

    const togglePlayPause = () => {
        setIsPlaying(prev => !prev);
    };

    const resetPlayback = () => {
        setTimeIndex(0);
        updateTrajectoryPosition(0);
    };

    const stepForward = () => {
        if (timeIndex < trajectory.length - 1) {
            setTimeIndex(prev => prev + 1);
            updateTrajectoryPosition(timeIndex + 1);
        }
    };

    const stepBackward = () => {
        if (timeIndex > 0) {
            setTimeIndex(prev => prev - 1);
            updateTrajectoryPosition(timeIndex - 1);
        }
    };

    const flyToImpact = () => {
        if (!map.current) return;
        
        map.current.flyTo({
            center: [impactPrediction.location.lon, impactPrediction.location.lat],
            zoom: 7,
            speed: 1.5,
            curve: 1.5,
            essential: true
        });
    };

    const flyToTrajectory = () => {
        if (!map.current) return;
        
        // Calculate bounds of trajectory
        const bounds = new mapboxgl.LngLatBounds();
        trajectory.forEach(point => {
            bounds.extend([point.position[0], point.position[1]]);
        });
        
        map.current.fitBounds(bounds, {
            padding: 60,
            maxZoom: 9,
            duration: 1500
        });
    };

    useEffect(() => {
        if (!mapContainer.current) return;
        
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: `mapbox://styles/mapbox/${mapStyle}`,
            center: [impactPrediction.location.lon, impactPrediction.location.lat],
            zoom: 5,
            pitch: viewMode === '3d' ? 60 : 0
        });
        
        map.current.on('load', () => {
            createMapControls();
            
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
                    'line-color': theme.palette.primary.main,
                    'line-width': 3,
                    'line-opacity': 0.8,
                    'line-gradient': [
                        'interpolate',
                        ['linear'],
                        ['line-progress'],
                        0, '#4CAF50',
                        0.5, '#FFC107',
                        1, '#F44336'
                    ]
                }
            });
            
            // Add impact point
            map.current?.addSource('impact', {
                type: 'geojson',
                data: {
                    type: 'Feature',
                    properties: {
                        title: 'Predicted Impact',
                        description: `Confidence: ${(impactPrediction.confidence * 100).toFixed(1)}%`
                    },
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
                    'circle-radius': 10,
                    'circle-color': '#F44336',
                    'circle-opacity': 0.8,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#FFFFFF'
                }
            });

            // Add popup on impact point click
            const popup = new mapboxgl.Popup({
                closeButton: false,
                closeOnClick: false,
                maxWidth: '300px',
                className: 'impact-popup'
            });

            map.current?.on('mouseenter', 'impact-point', (e) => {
                if (!map.current || !e.features) return;
                
                map.current.getCanvas().style.cursor = 'pointer';
                
                const coordinates: [number, number] = [
                    impactPrediction.location.lon,
                    impactPrediction.location.lat
                ];
                
                const popupContent = `
                    <h4>Impact Prediction</h4>
                    <div>Time: ${formatDateTimeString(impactPrediction.time)}</div>
                    <div>Coordinates: ${impactPrediction.location.lat.toFixed(4)}°N, ${impactPrediction.location.lon.toFixed(4)}°E</div>
                    <div>Velocity: ${impactPrediction.velocity.magnitude.toFixed(2)} m/s</div>
                    <div>Uncertainty: ±${impactPrediction.uncertainty_radius_km.toFixed(1)} km</div>
                    <div>Confidence: ${(impactPrediction.confidence * 100).toFixed(1)}%</div>
                `;
                
                popup.setLngLat(coordinates)
                    .setHTML(popupContent)
                    .addTo(map.current);
            });
            
            map.current?.on('mouseleave', 'impact-point', () => {
                if (!map.current) return;
                map.current.getCanvas().style.cursor = '';
                popup.remove();
            });
            
            // Add uncertainty circle
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
                    'fill-color': '#F44336',
                    'fill-opacity': 0.2,
                    'fill-outline-color': '#F44336'
                },
                layout: {
                    visibility: showUncertainty ? 'visible' : 'none'
                }
            });

            // Add uncertainty border
            map.current?.addLayer({
                id: 'uncertainty-border',
                type: 'line',
                source: 'uncertainty',
                paint: {
                    'line-color': '#F44336',
                    'line-width': 2,
                    'line-dasharray': [3, 3]
                },
                layout: {
                    visibility: showUncertainty ? 'visible' : 'none'
                }
            });
            
            // Add current position marker
            map.current?.addSource('current-position', {
                type: 'geojson',
                data: {
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'Point',
                        coordinates: [
                            trajectory[0].position[0],
                            trajectory[0].position[1],
                            trajectory[0].position[2]
                        ]
                    }
                }
            });
            
            map.current?.addLayer({
                id: 'current-position-point',
                type: 'circle',
                source: 'current-position',
                paint: {
                    'circle-radius': 8,
                    'circle-color': '#2196F3',
                    'circle-opacity': 1,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#FFFFFF'
                }
            });

            // Add pulse animation for current position
            map.current?.addLayer({
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
                    'circle-color': '#2196F3',
                    'circle-opacity': [
                        'interpolate',
                        ['linear'],
                        ['get', 'pulse'],
                        0, 0.6,
                        1, 0
                    ],
                    'circle-stroke-width': 0
                }
            });
            
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
                                time: point.time,
                                description: `Altitude: ${point.altitude.toFixed(1)} km, Fragments: ${point.fragments}`
                            },
                            geometry: {
                                type: 'Point',
                                coordinates: [
                                    trajectory[Math.min(index * 20, trajectory.length - 1)].position[0],
                                    trajectory[Math.min(index * 20, trajectory.length - 1)].position[1]
                                ]
                            }
                        }))
                    }
                });
                
                // Add circle for breakup points instead of using a symbol that may not be available
                map.current?.addLayer({
                    id: 'breakup-points',
                    type: 'circle',
                    source: 'breakups',
                    layout: {
                        'visibility': showBreakups ? 'visible' : 'none'
                    },
                    paint: {
                        'circle-radius': 10,
                        'circle-color': '#FF9800',
                        'circle-opacity': 0.9,
                        'circle-stroke-width': 2,
                        'circle-stroke-color': '#FFFFFF'
                    }
                });
                
                // Add text for fragment count
                map.current?.addLayer({
                    id: 'breakup-labels',
                    type: 'symbol',
                    source: 'breakups',
                    layout: {
                        'text-field': ['get', 'fragments'],
                        'text-font': ['Open Sans Bold'],
                        'text-size': 12,
                        'text-allow-overlap': true,
                        'visibility': showBreakups ? 'visible' : 'none'
                    },
                    paint: {
                        'text-color': '#FFFFFF',
                        'text-halo-color': '#000000',
                        'text-halo-width': 1
                    }
                });

                // Add popup for breakup points
                map.current?.on('mouseenter', 'breakup-points', (e) => {
                    if (!map.current || !e.features || !e.features[0]) return;
                    
                    map.current.getCanvas().style.cursor = 'pointer';
                    
                    const coordinates: [number, number] = e.features[0].geometry.type === 'Point' 
                        ? (e.features[0].geometry.coordinates as [number, number]) 
                        : [0, 0];
                    
                    const properties = e.features[0].properties;
                    if (!properties) return;
                    
                    const popupContent = `
                        <h4>Breakup Event</h4>
                        <div>Time: ${formatDateTimeString(properties.time as string)}</div>
                        <div>Altitude: ${Number(properties.altitude).toFixed(1)} km</div>
                        <div>Fragments: ${properties.fragments}</div>
                    `;
                    
                    popup.setLngLat(coordinates)
                        .setHTML(popupContent)
                        .addTo(map.current);
                });
                
                map.current?.on('mouseleave', 'breakup-points', () => {
                    if (!map.current) return;
                    map.current.getCanvas().style.cursor = '';
                    popup.remove();
                });
            }

            // Set current point for data display
            setCurrentPoint(trajectory[0]);
        });
        
        return () => {
            map.current?.remove();
        };
    }, [mapStyle]);

    // Update layers visibility when toggles change
    useEffect(() => {
        if (!map.current) return;
        
        if (map.current.getLayer('uncertainty-area')) {
            map.current.setLayoutProperty(
                'uncertainty-area',
                'visibility',
                showUncertainty ? 'visible' : 'none'
            );
        }
        
        if (map.current.getLayer('uncertainty-border')) {
            map.current.setLayoutProperty(
                'uncertainty-border',
                'visibility',
                showUncertainty ? 'visible' : 'none'
            );
        }
        
        if (map.current.getLayer('breakup-points')) {
            map.current.setLayoutProperty(
                'breakup-points',
                'visibility',
                showBreakups ? 'visible' : 'none'
            );
        }
        
        if (map.current.getLayer('breakup-labels')) {
            map.current.setLayoutProperty(
                'breakup-labels',
                'visibility',
                showBreakups ? 'visible' : 'none'
            );
        }
    }, [showUncertainty, showBreakups]);

    // Update view mode (2D/3D)
    useEffect(() => {
        if (!map.current) return;
        
        map.current.easeTo({
            pitch: viewMode === '3d' ? 60 : 0,
            duration: 1000
        });
    }, [viewMode]);

    // Handle playback animation
    useEffect(() => {
        if (isPlaying) {
            let lastTime = 0;
            let accumulatedTime = 0;
            const frameDuration = 1000 / (10 * playbackSpeed); // frames per second, adjusted by speed

            const animate = (time: number) => {
                if (!lastTime) lastTime = time;
                const deltaTime = time - lastTime;
                lastTime = time;
                
                accumulatedTime += deltaTime;
                
                if (accumulatedTime > frameDuration) {
                    accumulatedTime = 0;
                    
                    if (timeIndex < trajectory.length - 1) {
                        setTimeIndex(prev => prev + 1);
                        updateTrajectoryPosition(timeIndex + 1);
                    } else {
                        setIsPlaying(false);
                    }
                }
                
                if (isPlaying) {
                    animationRef.current = requestAnimationFrame(animate);
                }
            };
            
            animationRef.current = requestAnimationFrame(animate);
            
            return () => {
                if (animationRef.current) {
                    cancelAnimationFrame(animationRef.current);
                }
            };
        }
    }, [isPlaying, timeIndex, playbackSpeed, trajectory.length]);
    
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
            type: 'Feature' as const,
            properties: {},
            geometry: {
                type: 'Polygon' as const,
                coordinates: [coords]
            }
        };
    };
    
    const updateTrajectoryPosition = (index: number) => {
        if (!map.current) return;
        
        const point = trajectory[index];
        map.current.panTo([point.position[0], point.position[1]], { duration: 50 });
        
        // Update current position marker
        const source = map.current.getSource('current-position') as mapboxgl.GeoJSONSource;
        if (source) {
            source.setData({
                type: 'Feature',
                properties: {
                    pulse: (Date.now() % 1000) / 1000 // For pulse animation
                },
                geometry: {
                    type: 'Point',
                    coordinates: [point.position[0], point.position[1], point.position[2]]
                }
            });
        }
        
        // Update current point data
        setCurrentPoint(point);
    };

    // Format the altitude in km or m based on value
    const formatAltitude = (altitude: number) => {
        if (altitude >= 1000) {
            return `${(altitude / 1000).toFixed(2)} km`;
        } else {
            return `${altitude.toFixed(2)} m`;
        }
    };

    return (
        <Box sx={{ position: 'relative' }}>
            <Grid container spacing={2}>
                <Grid item xs={12}>
                    <ControlPanel>
                        <Grid container spacing={2} alignItems="center">
                            <Grid item xs={12} md={6}>
                                <Tabs
                                    value={tabValue}
                                    onChange={handleTabChange}
                                    aria-label="view options"
                                    variant="fullWidth"
                                    sx={{
                                        borderBottom: 1,
                                        borderColor: 'divider',
                                        mb: 1
                                    }}
                                >
                                    <Tab 
                                        icon={<MapIcon fontSize="small" />} 
                                        iconPosition="start"
                                        label="Map" 
                                    />
                                    <Tab 
                                        icon={<ViewInAr fontSize="small" />} 
                                        iconPosition="start"
                                        label="View Options" 
                                    />
                                    <Tab 
                                        icon={<Speed fontSize="small" />} 
                                        iconPosition="start"
                                        label="Controls" 
                                    />
                                </Tabs>
                                
                                <TabPanel value={tabValue} index={0}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Map Style</InputLabel>
                                        <Select
                                            value={mapStyle}
                                            onChange={(e) => setMapStyle(e.target.value)}
                                            label="Map Style"
                                        >
                                            {mapStyles.map(style => (
                                                <MenuItem key={style.id} value={style.id}>
                                                    {style.name}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </TabPanel>
                                
                                <TabPanel value={tabValue} index={1}>
                                    <Grid container spacing={2}>
                                        <Grid item xs={6}>
                                            <FormControl fullWidth size="small">
                                                <InputLabel>View Mode</InputLabel>
                                                <Select
                                                    value={viewMode}
                                                    onChange={(e) => setViewMode(e.target.value as '2d' | '3d')}
                                                    label="View Mode"
                                                >
                                                    <MenuItem value="2d">2D View</MenuItem>
                                                    <MenuItem value="3d">3D View</MenuItem>
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                        <Grid item xs={6} sx={{ display: 'flex', gap: 1 }}>
                                            <FormControlLabel
                                                control={
                                                    <Switch
                                                        size="small"
                                                        checked={showUncertainty}
                                                        onChange={(e) => setShowUncertainty(e.target.checked)}
                                                    />
                                                }
                                                label="Uncertainty"
                                            />
                                            <FormControlLabel
                                                control={
                                                    <Switch
                                                        size="small"
                                                        checked={showBreakups}
                                                        onChange={(e) => setShowBreakups(e.target.checked)}
                                                    />
                                                }
                                                label="Breakups"
                                            />
                                        </Grid>
                                    </Grid>
                                </TabPanel>
                                
                                <TabPanel value={tabValue} index={2}>
                                    <ButtonGroup variant="outlined" fullWidth>
                                        <Tooltip title="View entire trajectory">
                                            <Button onClick={flyToTrajectory}>
                                                <ZoomOut fontSize="small" />
                                            </Button>
                                        </Tooltip>
                                        <Tooltip title="Focus on impact point">
                                            <Button onClick={flyToImpact}>
                                                <MyLocation fontSize="small" />
                                            </Button>
                                        </Tooltip>
                                        <Tooltip title="Focus on current point">
                                            <Button onClick={() => updateTrajectoryPosition(timeIndex)}>
                                                <ZoomIn fontSize="small" />
                                            </Button>
                                        </Tooltip>
                                        <Tooltip title="Reset playback">
                                            <Button onClick={resetPlayback}>
                                                <RestartAlt fontSize="small" />
                                            </Button>
                                        </Tooltip>
                                    </ButtonGroup>
                                </TabPanel>
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                    <Typography variant="body2" sx={{ minWidth: '80px' }}>
                                        Time: {timeIndex} / {trajectory.length - 1}
                                    </Typography>
                                    <Chip 
                                        label={currentPoint ? `Alt: ${formatAltitude(calculateAltitude(currentPoint.position))}` : 'N/A'} 
                                        size="small" 
                                        color="primary" 
                                        variant="outlined" 
                                    />
                                    <Chip 
                                        label={currentPoint ? `Speed: ${Math.sqrt(
                                            currentPoint.velocity[0]**2 + 
                                            currentPoint.velocity[1]**2 + 
                                            currentPoint.velocity[2]**2
                                        ).toFixed(1)} m/s` : 'N/A'} 
                                        size="small" 
                                        color="secondary" 
                                        variant="outlined" 
                                    />
                                </Box>
                                <Slider
                                    value={timeIndex}
                                    onChange={(_, value) => {
                                        setTimeIndex(value as number);
                                        updateTrajectoryPosition(value as number);
                                    }}
                                    min={0}
                                    max={trajectory.length - 1}
                                    step={1}
                                    marks={breakupPoints ? breakupPoints.map((_, i) => ({
                                        value: Math.min(i * 20, trajectory.length - 1),
                                        label: <Warning fontSize="small" color="warning" />
                                    })) : undefined}
                                    valueLabelDisplay="auto"
                                    valueLabelFormat={value => `T+${value*10}s`}
                                />
                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
                                    <Typography variant="body2">Speed:</Typography>
                                    <Slider
                                        value={playbackSpeed}
                                        onChange={(_, value) => setPlaybackSpeed(value as number)}
                                        min={0.5}
                                        max={5}
                                        step={0.5}
                                        marks={[
                                            { value: 0.5, label: '0.5x' },
                                            { value: 1, label: '1x' },
                                            { value: 5, label: '5x' }
                                        ]}
                                        sx={{ minWidth: 120 }}
                                    />
                                </Stack>
                            </Grid>
                        </Grid>
                    </ControlPanel>
                </Grid>
                
                <Grid item xs={12} md={9}>
                    <MapContainer ref={mapContainer}>
                        <PlaybackControls>
                            <IconButton size="small" onClick={stepBackward}>
                                <SkipPrevious />
                            </IconButton>
                            <IconButton color="primary" onClick={togglePlayPause}>
                                {isPlaying ? <Pause /> : <PlayArrow />}
                            </IconButton>
                            <IconButton size="small" onClick={stepForward}>
                                <SkipNext />
                            </IconButton>
                        </PlaybackControls>
                        
                        <MapControls>
                            <ButtonGroup orientation="vertical" size="small">
                                <Tooltip title={viewMode === '3d' ? 'Switch to 2D' : 'Switch to 3D'} placement="left">
                                    <Button
                                        variant={viewMode === '3d' ? 'contained' : 'outlined'}
                                        onClick={() => setViewMode(viewMode === '3d' ? '2d' : '3d')}
                                    >
                                        <ViewInAr fontSize="small" />
                                    </Button>
                                </Tooltip>
                                <Tooltip title={showUncertainty ? 'Hide uncertainty' : 'Show uncertainty'} placement="left">
                                    <Button
                                        variant={showUncertainty ? 'contained' : 'outlined'}
                                        onClick={() => setShowUncertainty(!showUncertainty)}
                                        color="warning"
                                    >
                                        {showUncertainty ? <Visibility fontSize="small" /> : <VisibilityOff fontSize="small" />}
                                    </Button>
                                </Tooltip>
                                <Tooltip title="Focus on impact point" placement="left">
                                    <Button onClick={flyToImpact} color="error">
                                        <MyLocation fontSize="small" />
                                    </Button>
                                </Tooltip>
                            </ButtonGroup>
                        </MapControls>
                    </MapContainer>
                </Grid>
                
                <Grid item xs={12} md={3}>
                    <DataPanel>
                        <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ 
                                display: 'flex', 
                                alignItems: 'center',
                                borderBottom: `1px solid ${theme.palette.divider}`,
                                pb: 1
                            }}>
                                <Warning color="error" sx={{ mr: 1 }} />
                                Impact Prediction
                            </Typography>
                            
                            <DataItem>
                                <Typography variant="body2" color="text.secondary">Time:</Typography>
                                <Typography variant="body2" fontWeight="medium">
                                    {formatDateTimeString(impactPrediction.time)}
                                </Typography>
                            </DataItem>
                            
                            <DataItem>
                                <Typography variant="body2" color="text.secondary">Location:</Typography>
                                <Typography variant="body2" fontWeight="medium">
                                    {impactPrediction.location.lat.toFixed(4)}°N, {impactPrediction.location.lon.toFixed(4)}°E
                                </Typography>
                            </DataItem>
                            
                            <DataItem>
                                <Typography variant="body2" color="text.secondary">Velocity:</Typography>
                                <Typography variant="body2" fontWeight="medium">
                                    {impactPrediction.velocity.magnitude.toFixed(2)} m/s
                                </Typography>
                            </DataItem>
                            
                            <DataItem>
                                <Typography variant="body2" color="text.secondary">Uncertainty:</Typography>
                                <Typography variant="body2" fontWeight="medium">
                                    ±{impactPrediction.uncertainty_radius_km.toFixed(1)} km
                                </Typography>
                            </DataItem>
                            
                            <DataItem>
                                <Typography variant="body2" color="text.secondary">Confidence:</Typography>
                                <Typography variant="body2" fontWeight="medium">
                                    <Chip 
                                        label={`${(impactPrediction.confidence * 100).toFixed(1)}%`} 
                                        size="small" 
                                        color={impactPrediction.confidence > 0.8 ? "success" : impactPrediction.confidence > 0.6 ? "warning" : "error"} 
                                    />
                                </Typography>
                            </DataItem>
                            
                            {impactPrediction.monte_carlo_stats && (
                                <>
                                    <Divider sx={{ my: 2 }} />
                                    <Typography variant="subtitle2" sx={{ 
                                        fontWeight: 600, 
                                        display: 'flex', 
                                        alignItems: 'center', 
                                        mb: 1 
                                    }}>
                                        Monte Carlo Statistics
                                    </Typography>
                                    
                                    <DataItem>
                                        <Typography variant="body2" color="text.secondary">Samples:</Typography>
                                        <Typography variant="body2" fontWeight="medium">
                                            {impactPrediction.monte_carlo_stats.samples}
                                        </Typography>
                                    </DataItem>
                                    
                                    <DataItem>
                                        <Typography variant="body2" color="text.secondary">Time Std:</Typography>
                                        <Typography variant="body2" fontWeight="medium">
                                            ±{impactPrediction.monte_carlo_stats.time_std.toFixed(1)} s
                                        </Typography>
                                    </DataItem>
                                    
                                    <DataItem>
                                        <Typography variant="body2" color="text.secondary">Position Std:</Typography>
                                        <Typography variant="body2" fontWeight="medium">
                                            ±{impactPrediction.monte_carlo_stats.position_std.toFixed(1)} m
                                        </Typography>
                                    </DataItem>
                                    
                                    <DataItem>
                                        <Typography variant="body2" color="text.secondary">Velocity Std:</Typography>
                                        <Typography variant="body2" fontWeight="medium">
                                            ±{impactPrediction.monte_carlo_stats.velocity_std.toFixed(1)} m/s
                                        </Typography>
                                    </DataItem>
                                </>
                            )}
                            
                            {breakupPoints && breakupPoints.length > 0 && (
                                <>
                                    <Divider sx={{ my: 2 }} />
                                    <Typography variant="subtitle2" sx={{ 
                                        fontWeight: 600, 
                                        display: 'flex', 
                                        alignItems: 'center',
                                        mb: 1
                                    }}>
                                        <Warning color="warning" sx={{ mr: 1, fontSize: '1.1rem' }} />
                                        Breakup Events ({breakupPoints.length})
                                    </Typography>
                                    
                                    {breakupPoints.map((point, index) => (
                                        <Box key={index} sx={{ 
                                            mb: 1, 
                                            pb: 1, 
                                            borderBottom: index !== breakupPoints.length - 1 ? `1px dashed ${theme.palette.divider}` : 'none' 
                                        }}>
                                            <DataItem>
                                                <Typography variant="body2" color="text.secondary">Time:</Typography>
                                                <Typography variant="body2" fontWeight="medium">
                                                    {formatDateTimeString(point.time)}
                                                </Typography>
                                            </DataItem>
                                            <DataItem>
                                                <Typography variant="body2" color="text.secondary">Altitude:</Typography>
                                                <Typography variant="body2" fontWeight="medium">
                                                    {point.altitude.toFixed(1)} km
                                                </Typography>
                                            </DataItem>
                                            <DataItem>
                                                <Typography variant="body2" color="text.secondary">Fragments:</Typography>
                                                <Typography variant="body2" fontWeight="medium">
                                                    <Chip 
                                                        label={point.fragments} 
                                                        size="small" 
                                                        color="warning" 
                                                    />
                                                </Typography>
                                            </DataItem>
                                        </Box>
                                    ))}
                                </>
                            )}
                        </CardContent>
                    </DataPanel>
                </Grid>
            </Grid>
        </Box>
    );
};

export default TrajectoryViewer; 