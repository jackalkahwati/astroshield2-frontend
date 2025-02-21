import React, { useState, useEffect } from 'react';
import { 
    Box,
    Paper,
    Typography,
    Grid,
    Button,
    TextField,
    CircularProgress,
    Alert,
    FormControl,
    InputLabel,
    Select,
    MenuItem
} from '@mui/material';
import { styled } from '@mui/material/styles';
import TrajectoryViewer from '../components/TrajectoryViewer';

const ConfigPanel = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    marginBottom: theme.spacing(2)
}));

interface TrajectoryConfig {
    atmospheric_model: string;
    wind_model: string;
    monte_carlo_samples: number;
    object_properties: {
        mass: number;
        area: number;
        cd: number;
    };
    breakup_model: {
        enabled: boolean;
        fragment_count: number;
        mass_distribution: string;
        velocity_perturbation: number;
    };
}

const defaultConfig: TrajectoryConfig = {
    atmospheric_model: 'nrlmsise',
    wind_model: 'hwm14',
    monte_carlo_samples: 1000,
    object_properties: {
        mass: 1000,
        area: 1.0,
        cd: 2.2
    },
    breakup_model: {
        enabled: false,
        fragment_count: 10,
        mass_distribution: 'log_normal',
        velocity_perturbation: 100.0
    }
};

const TrajectoryAnalysis: React.FC = () => {
    const [config, setConfig] = useState<TrajectoryConfig>(defaultConfig);
    const [initialState, setInitialState] = useState({
        position: { x: 0, y: 0, z: 6471000 }, // 100km altitude
        velocity: { x: 1000, y: 0, z: -1000 }  // Example reentry velocity
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [trajectoryData, setTrajectoryData] = useState<any>(null);
    
    const handleConfigChange = (field: string, value: any) => {
        setConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };
    
    const handleObjectPropertyChange = (field: string, value: number) => {
        setConfig(prev => ({
            ...prev,
            object_properties: {
                ...prev.object_properties,
                [field]: value
            }
        }));
    };
    
    const handleBreakupConfigChange = (field: string, value: any) => {
        setConfig(prev => ({
            ...prev,
            breakup_model: {
                ...prev.breakup_model,
                [field]: value
            }
        }));
    };
    
    const handleInitialStateChange = (component: string, axis: string, value: number) => {
        setInitialState(prev => ({
            ...prev,
            [component]: {
                ...prev[component as keyof typeof prev],
                [axis]: value
            }
        }));
    };
    
    const runAnalysis = async () => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch('/api/trajectory/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    config,
                    initial_state: [
                        initialState.position.x,
                        initialState.position.y,
                        initialState.position.z,
                        initialState.velocity.x,
                        initialState.velocity.y,
                        initialState.velocity.z
                    ]
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to analyze trajectory');
            }
            
            const data = await response.json();
            setTrajectoryData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
                Trajectory Analysis
            </Typography>
            
            <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                    <ConfigPanel>
                        <Typography variant="h6" gutterBottom>
                            Configuration
                        </Typography>
                        
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <FormControl fullWidth>
                                    <InputLabel>Atmospheric Model</InputLabel>
                                    <Select
                                        value={config.atmospheric_model}
                                        onChange={(e) => handleConfigChange('atmospheric_model', e.target.value)}
                                    >
                                        <MenuItem value="exponential">Exponential</MenuItem>
                                        <MenuItem value="nrlmsise">NRLMSISE-00</MenuItem>
                                        <MenuItem value="jacchia">Jacchia</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            
                            <Grid item xs={12}>
                                <FormControl fullWidth>
                                    <InputLabel>Wind Model</InputLabel>
                                    <Select
                                        value={config.wind_model}
                                        onChange={(e) => handleConfigChange('wind_model', e.target.value)}
                                    >
                                        <MenuItem value="custom">Basic</MenuItem>
                                        <MenuItem value="hwm14">HWM14</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="Monte Carlo Samples"
                                    type="number"
                                    value={config.monte_carlo_samples}
                                    onChange={(e) => handleConfigChange('monte_carlo_samples', parseInt(e.target.value))}
                                />
                            </Grid>
                        </Grid>
                        
                        <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                            Object Properties
                        </Typography>
                        
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="Mass (kg)"
                                    type="number"
                                    value={config.object_properties.mass}
                                    onChange={(e) => handleObjectPropertyChange('mass', parseFloat(e.target.value))}
                                />
                            </Grid>
                            
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="Area (m²)"
                                    type="number"
                                    value={config.object_properties.area}
                                    onChange={(e) => handleObjectPropertyChange('area', parseFloat(e.target.value))}
                                />
                            </Grid>
                            
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="Drag Coefficient"
                                    type="number"
                                    value={config.object_properties.cd}
                                    onChange={(e) => handleObjectPropertyChange('cd', parseFloat(e.target.value))}
                                />
                            </Grid>
                        </Grid>
                        
                        <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                            Breakup Model
                        </Typography>
                        
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <FormControl fullWidth>
                                    <InputLabel>Enable Breakup</InputLabel>
                                    <Select
                                        value={config.breakup_model.enabled}
                                        onChange={(e) => handleBreakupConfigChange('enabled', e.target.value)}
                                    >
                                        <MenuItem value={true}>Yes</MenuItem>
                                        <MenuItem value={false}>No</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            
                            {config.breakup_model.enabled && (
                                <>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Fragment Count"
                                            type="number"
                                            value={config.breakup_model.fragment_count}
                                            onChange={(e) => handleBreakupConfigChange('fragment_count', parseInt(e.target.value))}
                                        />
                                    </Grid>
                                    
                                    <Grid item xs={12}>
                                        <FormControl fullWidth>
                                            <InputLabel>Mass Distribution</InputLabel>
                                            <Select
                                                value={config.breakup_model.mass_distribution}
                                                onChange={(e) => handleBreakupConfigChange('mass_distribution', e.target.value)}
                                            >
                                                <MenuItem value="log_normal">Log Normal</MenuItem>
                                                <MenuItem value="equal">Equal</MenuItem>
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Velocity Perturbation (m/s)"
                                            type="number"
                                            value={config.breakup_model.velocity_perturbation}
                                            onChange={(e) => handleBreakupConfigChange('velocity_perturbation', parseFloat(e.target.value))}
                                        />
                                    </Grid>
                                </>
                            )}
                        </Grid>
                        
                        <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                            Initial State
                        </Typography>
                        
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <Typography variant="subtitle2">Position (m)</Typography>
                                <Grid container spacing={1}>
                                    {['x', 'y', 'z'].map(axis => (
                                        <Grid item xs={4} key={`pos-${axis}`}>
                                            <TextField
                                                fullWidth
                                                label={axis.toUpperCase()}
                                                type="number"
                                                value={initialState.position[axis as keyof typeof initialState.position]}
                                                onChange={(e) => handleInitialStateChange('position', axis, parseFloat(e.target.value))}
                                            />
                                        </Grid>
                                    ))}
                                </Grid>
                            </Grid>
                            
                            <Grid item xs={12}>
                                <Typography variant="subtitle2">Velocity (m/s)</Typography>
                                <Grid container spacing={1}>
                                    {['x', 'y', 'z'].map(axis => (
                                        <Grid item xs={4} key={`vel-${axis}`}>
                                            <TextField
                                                fullWidth
                                                label={axis.toUpperCase()}
                                                type="number"
                                                value={initialState.velocity[axis as keyof typeof initialState.velocity]}
                                                onChange={(e) => handleInitialStateChange('velocity', axis, parseFloat(e.target.value))}
                                            />
                                        </Grid>
                                    ))}
                                </Grid>
                            </Grid>
                        </Grid>
                        
                        <Box sx={{ mt: 2 }}>
                            <Button
                                fullWidth
                                variant="contained"
                                color="primary"
                                onClick={runAnalysis}
                                disabled={loading}
                            >
                                {loading ? <CircularProgress size={24} /> : 'Run Analysis'}
                            </Button>
                        </Box>
                    </ConfigPanel>
                </Grid>
                
                <Grid item xs={12} md={8}>
                    {error && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {error}
                        </Alert>
                    )}
                    
                    {trajectoryData && (
                        <TrajectoryViewer
                            trajectory={trajectoryData.trajectory}
                            impactPrediction={trajectoryData.impact_prediction}
                            breakupPoints={trajectoryData.breakup_events}
                        />
                    )}
                </Grid>
            </Grid>
        </Box>
    );
};

export default TrajectoryAnalysis; 