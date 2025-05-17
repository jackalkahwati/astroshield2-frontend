import React, { useState, useEffect, useCallback } from 'react';
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
    MenuItem,
    Tabs,
    Tab,
    Divider,
    Tooltip,
    IconButton,
    Chip,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Snackbar,
    Card,
    CardContent,
    CardMedia,
    CardActionArea,
    LinearProgress,
    Badge,
    Avatar,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Stack,
    Switch,
    FormControlLabel,
    useTheme,
    alpha,
    AlertTitle
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
    Refresh, 
    PlayArrow, 
    Save, 
    InfoOutlined, 
    ExpandMore,
    SettingsOutlined,
    HistoryOutlined,
    TipsAndUpdatesOutlined,
    RocketLaunch,
    Science,
    Satellite,
    Warning,
    Speed,
    CheckCircle,
    CloudDownload,
    DeleteOutline,
    AutoGraph,
    AirplaneTicket,
    Article,
    FolderShared,
    Assessment,
    BarChart,
    LightMode,
    DarkMode,
    Favorite,
    FavoriteBorder
} from '@mui/icons-material';
import TrajectoryViewer from '../components/TrajectoryViewer';

const ConfigPanel = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(3),
    marginBottom: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
    height: '100%',
    boxShadow: theme.shadows[2],
    background: theme.palette.mode === 'dark' 
        ? alpha(theme.palette.background.paper, 0.8)
        : 'linear-gradient(to bottom, rgba(255,255,255,0.95), rgba(240,245,255,0.9))',
    backdropFilter: 'blur(8px)',
    border: `1px solid ${theme.palette.divider}`
}));

const StyledAccordion = styled(Accordion)(({ theme }) => ({
    boxShadow: 'none',
    backgroundColor: 'transparent',
    '&:before': {
        display: 'none',
    },
    '&.Mui-expanded': {
        margin: 0,
    },
    '& .MuiAccordionSummary-root.Mui-expanded': {
        minHeight: '48px',
        borderBottom: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
    },
    '& .MuiAccordionSummary-content.Mui-expanded': {
        margin: '12px 0'
    }
}));

const ScenarioCard = styled(Card)(({ theme }) => ({
    cursor: 'pointer',
    transition: theme.transitions.create(['transform', 'box-shadow'], {
        duration: theme.transitions.duration.standard,
    }),
    '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: theme.shadows[6]
    }
}));

const RunButton = styled(Button)(({ theme }) => ({
    marginTop: theme.spacing(3),
    borderRadius: theme.shape.borderRadius,
    padding: theme.spacing(1.5, 0),
    background: theme.palette.mode === 'dark'
        ? `linear-gradient(45deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`
        : `linear-gradient(45deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.light} 100%)`,
    boxShadow: `0 4px 10px ${alpha(theme.palette.primary.main, 0.25)}`,
    transition: theme.transitions.create(['background', 'box-shadow'], {
        duration: theme.transitions.duration.standard,
    }),
    '&:hover': {
        boxShadow: `0 6px 15px ${alpha(theme.palette.primary.main, 0.35)}`,
    }
}));

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

const FadeInBox = styled(Box)(({ theme }) => ({
    animation: 'fadeIn 0.4s ease-in-out',
    '@keyframes fadeIn': {
        '0%': {
            opacity: 0,
            transform: 'translateY(10px)',
        },
        '100%': {
            opacity: 1,
            transform: 'translateY(0)',
        },
    },
}));

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    const theme = useTheme();

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`trajectory-tabpanel-${index}`}
            aria-labelledby={`trajectory-tab-${index}`}
            {...other}
            style={{ 
                padding: theme.spacing(2, 0),
                height: '100%',
                overflow: 'auto'
            }}
        >
            {value === index && (
                <FadeInBox>{children}</FadeInBox>
            )}
        </div>
    );
}

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

// Predefined scenarios for quick setup
const predefinedScenarios = [
    {
        name: "Controlled Deorbit",
        description: "Typical scenario for planned spacecraft reentry",
        config: {
            atmospheric_model: "nrlmsise",
            wind_model: "hwm14",
            monte_carlo_samples: 1000,
            object_properties: {
                mass: 1200,
                area: 1.2,
                cd: 2.2
            },
            breakup_model: {
                enabled: true,
                fragment_count: 8,
                mass_distribution: "log_normal",
                velocity_perturbation: 50.0
            }
        },
        initialState: {
            position: { x: 0, y: 0, z: 6471000 },
            velocity: { x: 900, y: 100, z: -900 }
        }
    },
    {
        name: "Uncontrolled Reentry",
        description: "Unplanned atmospheric reentry of defunct satellite",
        config: {
            atmospheric_model: "nrlmsise",
            wind_model: "hwm14",
            monte_carlo_samples: 2000,
            object_properties: {
                mass: 800,
                area: 3.5,
                cd: 2.4
            },
            breakup_model: {
                enabled: true,
                fragment_count: 15,
                mass_distribution: "log_normal",
                velocity_perturbation: 120.0
            }
        },
        initialState: {
            position: { x: 200000, y: 100000, z: 6471000 },
            velocity: { x: 1200, y: -300, z: -700 }
        }
    },
    {
        name: "Rocket Body",
        description: "Upper stage or rocket body reentry",
        config: {
            atmospheric_model: "nrlmsise",
            wind_model: "hwm14",
            monte_carlo_samples: 1500,
            object_properties: {
                mass: 3500,
                area: 8.0,
                cd: 1.8
            },
            breakup_model: {
                enabled: true,
                fragment_count: 20,
                mass_distribution: "log_normal",
                velocity_perturbation: 150.0
            }
        },
        initialState: {
            position: { x: -100000, y: 200000, z: 6471000 },
            velocity: { x: 800, y: 500, z: -1100 }
        }
    }
];

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

// Mock data for development
const mockTrajectoryData = {
    trajectory: Array(100).fill(0).map((_, i) => ({
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
    })),
    impactPrediction: {
        time: new Date().toISOString(),
        location: {
            lat: 40.7128 + 1,
            lon: -74.0060 + 1
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
        confidence: 0.95,
        monte_carlo_stats: {
            samples: 1000,
            time_std: 5.2,
            position_std: 872.3,
            velocity_std: 12.8
        }
    },
    breakupPoints: [
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
    const [tabValue, setTabValue] = useState(0);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    
    // Add mock data state
    const [mockData] = useState(mockTrajectoryData);
    
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
    
    const loadScenario = (index: number) => {
        const scenario = predefinedScenarios[index];
        setConfig(scenario.config);
        setInitialState(scenario.initialState);
        setSuccessMessage(`Loaded scenario: ${scenario.name}`);
    };
    
    const resetConfig = () => {
        setConfig(defaultConfig);
        setInitialState({
            position: { x: 0, y: 0, z: 6471000 },
            velocity: { x: 1000, y: 0, z: -1000 }
        });
        setSuccessMessage("Configuration reset to defaults");
    };
    
    const BASE_API_URL = 'http://localhost:3002'; // Point to our FastAPI backend
    
    const runAnalysis = async () => {
        setLoading(true);
        setError(null);
        
        try {
            try {
                console.log("Calling trajectory analysis API endpoint...");
                const response = await fetch(`${BASE_API_URL}/api/trajectory/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
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
                
                if (response.ok) {
                    const data = await response.json();
                    console.log("API response received:", data);
                    setTrajectoryData(data);
                    setSuccessMessage("Analysis completed successfully");
                    return;
                } else {
                    const errorText = await response.text();
                    console.error("API error:", response.status, errorText);
                    throw new Error(`API returned error ${response.status}: ${errorText}`);
                }
            } catch (apiError) {
                console.error("API call failed, falling back to mock data:", apiError);
                // If API call fails, fall back to mock data
                console.log("Using mock trajectory data as fallback");
                await new Promise(resolve => setTimeout(resolve, 1000)); // Short delay for UX
                setTrajectoryData(mockData);
                setSuccessMessage("Analysis completed with simulated data (API not available)");
            }
        } catch (err) {
            console.error("Analysis error:", err);
            setError(err instanceof Error ? err.message : 'Failed to analyze trajectory');
        } finally {
            setLoading(false);
        }
    };
    
    const saveAnalysis = () => {
        // This would save the current configuration and results
        setSuccessMessage("Analysis configuration saved");
    };
    
    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };
    
    const handleCloseSnackbar = () => {
        setSuccessMessage(null);
    };
    
    return (
        <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                    <ConfigPanel>
                        <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'space-between',
                            mb: 2
                        }}>
                            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
                                <SettingsOutlined sx={{ mr: 1 }} /> Configuration
                            </Typography>
                            <Box>
                                <Tooltip title="Reset to defaults">
                                    <IconButton onClick={resetConfig} size="small">
                                        <Refresh />
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title="Save configuration">
                                    <IconButton onClick={saveAnalysis} size="small">
                                        <Save />
                                    </IconButton>
                                </Tooltip>
                            </Box>
                        </Box>
                        
                        <Tabs
                            value={tabValue}
                            onChange={handleTabChange}
                            variant="fullWidth"
                            aria-label="trajectory configuration tabs"
                            sx={{ 
                                mb: 2,
                                '& .MuiTab-root': {
                                    textTransform: 'none',
                                    minHeight: '48px',
                                }
                            }}
                        >
                            <Tab 
                                icon={<TipsAndUpdatesOutlined fontSize="small" />} 
                                iconPosition="start"
                                label="Quick Setup" 
                                id="trajectory-tab-0" 
                                aria-controls="trajectory-tabpanel-0" 
                            />
                            <Tab 
                                icon={<SettingsOutlined fontSize="small" />} 
                                iconPosition="start"
                                label="Advanced" 
                                id="trajectory-tab-1" 
                                aria-controls="trajectory-tabpanel-1" 
                            />
                            <Tab 
                                icon={<HistoryOutlined fontSize="small" />} 
                                iconPosition="start"
                                label="History" 
                                id="trajectory-tab-2" 
                                aria-controls="trajectory-tabpanel-2" 
                            />
                        </Tabs>
                        
                        <TabPanel value={tabValue} index={0}>
                            <Typography variant="body2" sx={{ mb: 3, fontWeight: 500 }}>
                                Choose a predefined scenario for quick configuration:
                            </Typography>
                            
                            <Grid container spacing={2}>
                                {predefinedScenarios.map((scenario, index) => {
                                    // Select an icon based on the scenario type
                                    const getScenarioIcon = () => {
                                        switch(index) {
                                            case 0: return <Satellite color="primary" />;
                                            case 1: return <Warning color="warning" />;
                                            case 2: return <RocketLaunch color="error" />;
                                            default: return <Science color="info" />;
                                        }
                                    };
                                    
                                    // Get background color based on scenario type
                                    const getScenarioColor = (theme: any) => {
                                        switch(index) {
                                            case 0: return alpha(theme.palette.primary.main, 0.08);
                                            case 1: return alpha(theme.palette.warning.main, 0.08);
                                            case 2: return alpha(theme.palette.error.main, 0.08);
                                            default: return alpha(theme.palette.info.main, 0.08);
                                        }
                                    };
                                    
                                    return (
                                        <Grid item xs={12} sm={6} md={12} lg={6} key={index}>
                                            <ScenarioCard 
                                                elevation={2} 
                                                onClick={() => loadScenario(index)}
                                                sx={(theme) => ({
                                                    position: 'relative',
                                                    overflow: 'visible',
                                                    background: getScenarioColor(theme)
                                                })}
                                            >
                                                <Box 
                                                    sx={{ 
                                                        position: 'absolute', 
                                                        top: -16, 
                                                        left: 20, 
                                                        padding: '8px', 
                                                        borderRadius: '50%', 
                                                        bgcolor: 'background.paper',
                                                        boxShadow: 2,
                                                        zIndex: 1
                                                    }}
                                                >
                                                    <Avatar 
                                                        sx={{ 
                                                            width: 40, 
                                                            height: 40,
                                                            bgcolor: (theme) => 
                                                                index === 0 ? theme.palette.primary.main :
                                                                index === 1 ? theme.palette.warning.main :
                                                                theme.palette.error.main
                                                        }}
                                                    >
                                                        {getScenarioIcon()}
                                                    </Avatar>
                                                </Box>
                                                
                                                <CardContent sx={{ pt: 4 }}>
                                                    <Typography variant="h6" fontWeight="600" gutterBottom>
                                                        {scenario.name}
                                                    </Typography>
                                                    
                                                    <Typography variant="body2" color="text.secondary" paragraph sx={{ minHeight: '40px' }}>
                                                        {scenario.description}
                                                    </Typography>
                                                    
                                                    <Grid container spacing={1} sx={{ mt: 1 }}>
                                                        <Grid item xs={6}>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Object Mass
                                                                </Typography>
                                                                <Typography variant="body2" fontWeight="medium">
                                                                    {scenario.config.object_properties.mass} kg
                                                                </Typography>
                                                            </Box>
                                                        </Grid>
                                                        
                                                        <Grid item xs={6}>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Drag Coefficient
                                                                </Typography>
                                                                <Typography variant="body2" fontWeight="medium">
                                                                    {scenario.config.object_properties.cd}
                                                                </Typography>
                                                            </Box>
                                                        </Grid>
                                                        
                                                        <Grid item xs={6}>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Area
                                                                </Typography>
                                                                <Typography variant="body2" fontWeight="medium">
                                                                    {scenario.config.object_properties.area} m²
                                                                </Typography>
                                                            </Box>
                                                        </Grid>
                                                        
                                                        <Grid item xs={6}>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Fragments
                                                                </Typography>
                                                                <Chip 
                                                                    label={scenario.config.breakup_model.fragment_count} 
                                                                    size="small" 
                                                                    color={
                                                                        index === 0 ? "primary" :
                                                                        index === 1 ? "warning" : 
                                                                        "error"
                                                                    }
                                                                    sx={{ width: 'fit-content' }}
                                                                />
                                                            </Box>
                                                        </Grid>
                                                    </Grid>
                                                    
                                                    <Box sx={{ 
                                                        display: 'flex', 
                                                        justifyContent: 'space-between', 
                                                        alignItems: 'center',
                                                        mt: 2,
                                                        pt: 1,
                                                        borderTop: '1px dashed',
                                                        borderColor: 'divider'
                                                    }}>
                                                        <Chip
                                                            label={scenario.config.atmospheric_model.toUpperCase()}
                                                            size="small"
                                                            variant="outlined"
                                                        />
                                                        <Tooltip title="Load this scenario">
                                                            <Button 
                                                                size="small" 
                                                                endIcon={<PlayArrow />}
                                                                color={
                                                                    index === 0 ? "primary" :
                                                                    index === 1 ? "warning" : 
                                                                    "error"
                                                                }
                                                            >
                                                                Select
                                                            </Button>
                                                        </Tooltip>
                                                    </Box>
                                                </CardContent>
                                            </ScenarioCard>
                                        </Grid>
                                    );
                                })}
                            </Grid>
                            
                            <RunButton
                                fullWidth
                                variant="contained"
                                color="primary"
                                onClick={runAnalysis}
                                disabled={loading}
                                startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
                                size="large"
                            >
                                {loading ? 'Running Analysis...' : 'Run Trajectory Analysis'}
                            </RunButton>
                            
                            {/* Analysis progress indicators */}
                            {loading && (
                                <Box sx={{ mt: 3 }}>
                                    <Typography variant="body2" fontWeight="medium" sx={{ mb: 1 }}>
                                        Analysis Progress
                                    </Typography>
                                    <LinearProgress 
                                        variant="indeterminate" 
                                        sx={{ 
                                            height: 8,
                                            borderRadius: 4,
                                            mb: 2
                                        }}
                                    />
                                    <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                                        <Chip 
                                            label={`Samples: ${config.monte_carlo_samples}`} 
                                            size="small" 
                                            variant="outlined"
                                            icon={<Assessment fontSize="small" />}
                                        />
                                        <Chip 
                                            label={`Model: ${config.atmospheric_model.toUpperCase()}`} 
                                            size="small" 
                                            variant="outlined"
                                            icon={<Science fontSize="small" />}
                                        />
                                    </Stack>
                                </Box>
                            )}
                        </TabPanel>
                        
                        <TabPanel value={tabValue} index={1}>
                            <StyledAccordion defaultExpanded>
                                <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Typography variant="subtitle1">Simulation Parameters</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <FormControl fullWidth size="small">
                                                <InputLabel>Atmospheric Model</InputLabel>
                                                <Select
                                                    value={config.atmospheric_model}
                                                    onChange={(e) => handleConfigChange('atmospheric_model', e.target.value)}
                                                    label="Atmospheric Model"
                                                >
                                                    <MenuItem value="exponential">Exponential</MenuItem>
                                                    <MenuItem value="nrlmsise">NRLMSISE-00</MenuItem>
                                                    <MenuItem value="jacchia">Jacchia</MenuItem>
                                                </Select>
                                            </FormControl>
                                        </Grid>
                                        
                                        <Grid item xs={12}>
                                            <FormControl fullWidth size="small">
                                                <InputLabel>Wind Model</InputLabel>
                                                <Select
                                                    value={config.wind_model}
                                                    onChange={(e) => handleConfigChange('wind_model', e.target.value)}
                                                    label="Wind Model"
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
                                                size="small"
                                                InputProps={{
                                                    endAdornment: (
                                                        <Tooltip title="More samples = higher precision but slower performance">
                                                            <InfoOutlined fontSize="small" sx={{ ml: 1, color: 'text.secondary' }} />
                                                        </Tooltip>
                                                    )
                                                }}
                                            />
                                        </Grid>
                                    </Grid>
                                </AccordionDetails>
                            </StyledAccordion>
                            
                            <StyledAccordion defaultExpanded>
                                <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Typography variant="subtitle1">Object Properties</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <TextField
                                                fullWidth
                                                label="Mass (kg)"
                                                type="number"
                                                value={config.object_properties.mass}
                                                onChange={(e) => handleObjectPropertyChange('mass', parseFloat(e.target.value))}
                                                size="small"
                                            />
                                        </Grid>
                                        
                                        <Grid item xs={12}>
                                            <TextField
                                                fullWidth
                                                label="Area (m²)"
                                                type="number"
                                                value={config.object_properties.area}
                                                onChange={(e) => handleObjectPropertyChange('area', parseFloat(e.target.value))}
                                                size="small"
                                            />
                                        </Grid>
                                        
                                        <Grid item xs={12}>
                                            <TextField
                                                fullWidth
                                                label="Drag Coefficient"
                                                type="number"
                                                value={config.object_properties.cd}
                                                onChange={(e) => handleObjectPropertyChange('cd', parseFloat(e.target.value))}
                                                size="small"
                                                InputProps={{
                                                    endAdornment: (
                                                        <Tooltip title="Typical values: 2.0-2.5 for satellites">
                                                            <InfoOutlined fontSize="small" sx={{ ml: 1, color: 'text.secondary' }} />
                                                        </Tooltip>
                                                    )
                                                }}
                                            />
                                        </Grid>
                                    </Grid>
                                </AccordionDetails>
                            </StyledAccordion>
                            
                            <StyledAccordion>
                                <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Typography variant="subtitle1">Breakup Model</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <FormControl fullWidth size="small">
                                                <InputLabel>Enable Breakup</InputLabel>
                                                <Select
                                                    value={config.breakup_model.enabled ? "true" : "false"}
                                                    onChange={(e) => handleBreakupConfigChange('enabled', e.target.value === "true")}
                                                    label="Enable Breakup"
                                                >
                                                    <MenuItem value="true">Yes</MenuItem>
                                                    <MenuItem value="false">No</MenuItem>
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
                                                        size="small"
                                                    />
                                                </Grid>
                                                
                                                <Grid item xs={12}>
                                                    <FormControl fullWidth size="small">
                                                        <InputLabel>Mass Distribution</InputLabel>
                                                        <Select
                                                            value={config.breakup_model.mass_distribution}
                                                            onChange={(e) => handleBreakupConfigChange('mass_distribution', e.target.value)}
                                                            label="Mass Distribution"
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
                                                        size="small"
                                                    />
                                                </Grid>
                                            </>
                                        )}
                                    </Grid>
                                </AccordionDetails>
                            </StyledAccordion>
                            
                            <StyledAccordion>
                                <AccordionSummary expandIcon={<ExpandMore />}>
                                    <Typography variant="subtitle1">Initial State</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <Typography variant="subtitle2" gutterBottom>Position (m)</Typography>
                                            <Grid container spacing={1}>
                                                {['x', 'y', 'z'].map(axis => (
                                                    <Grid item xs={4} key={`pos-${axis}`}>
                                                        <TextField
                                                            fullWidth
                                                            label={axis.toUpperCase()}
                                                            type="number"
                                                            value={initialState.position[axis as keyof typeof initialState.position]}
                                                            onChange={(e) => handleInitialStateChange('position', axis, parseFloat(e.target.value))}
                                                            size="small"
                                                        />
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </Grid>
                                        
                                        <Grid item xs={12}>
                                            <Typography variant="subtitle2" gutterBottom>Velocity (m/s)</Typography>
                                            <Grid container spacing={1}>
                                                {['x', 'y', 'z'].map(axis => (
                                                    <Grid item xs={4} key={`vel-${axis}`}>
                                                        <TextField
                                                            fullWidth
                                                            label={axis.toUpperCase()}
                                                            type="number"
                                                            value={initialState.velocity[axis as keyof typeof initialState.velocity]}
                                                            onChange={(e) => handleInitialStateChange('velocity', axis, parseFloat(e.target.value))}
                                                            size="small"
                                                        />
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </Grid>
                                    </Grid>
                                </AccordionDetails>
                            </StyledAccordion>
                            
                            <Button
                                fullWidth
                                variant="contained"
                                color="primary"
                                onClick={runAnalysis}
                                disabled={loading}
                                startIcon={loading ? <CircularProgress size={16} /> : <PlayArrow />}
                                size="large"
                                sx={{ mt: 3 }}
                            >
                                {loading ? 'Running Analysis...' : 'Run Analysis'}
                            </Button>
                        </TabPanel>
                        
                        <TabPanel value={tabValue} index={2}>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                Recent trajectory analyses:
                            </Typography>
                            
                            <Box sx={{ 
                                p: 3, 
                                display: 'flex', 
                                justifyContent: 'center', 
                                alignItems: 'center',
                                bgcolor: 'background.paper',
                                borderRadius: 1,
                                border: '1px dashed',
                                borderColor: 'divider'
                            }}>
                                <Typography variant="body2" color="text.secondary">
                                    No previous analyses found
                                </Typography>
                            </Box>
                        </TabPanel>
                    </ConfigPanel>
                </Grid>
                
                <Grid item xs={12} md={8}>
                    <Paper 
                        elevation={2} 
                        sx={{ 
                            height: '100%', 
                            display: 'flex', 
                            flexDirection: 'column',
                            borderRadius: 2,
                            overflow: 'hidden'
                        }}
                    >
                        {loading ? (
                            <Box sx={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                justifyContent: 'center', 
                                alignItems: 'center', 
                                p: 5,
                                flex: 1
                            }}>
                                <Box sx={{ position: 'relative', mb: 4 }}>
                                    <CircularProgress 
                                        size={80} 
                                        color="primary" 
                                        thickness={3}
                                        sx={{ 
                                            opacity: 0.3,
                                        }}
                                    />
                                    <CircularProgress 
                                        size={80} 
                                        color="secondary" 
                                        thickness={3}
                                        variant="determinate"
                                        value={70}
                                        sx={{ 
                                            position: 'absolute',
                                            top: 0,
                                            left: 0,
                                            opacity: 0.7,
                                        }}
                                    />
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            top: 0,
                                            left: 0,
                                            bottom: 0,
                                            right: 0,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                        }}
                                    >
                                        <RocketLaunch color="primary" fontSize="large" />
                                    </Box>
                                </Box>
                                <Typography variant="h5" fontWeight="600" sx={{ mb: 1 }}>
                                    Running Trajectory Analysis
                                </Typography>
                                <Typography variant="body1" sx={{ mb: 4 }}>
                                    This may take a few moments...
                                </Typography>
                                
                                <Box sx={{ width: '100%', maxWidth: 400 }}>
                                    <Stack spacing={2}>
                                        <Box sx={{ width: '100%' }}>
                                            <Typography variant="body2" fontWeight="500" sx={{ mb: 0.5, display: 'flex', justifyContent: 'space-between' }}>
                                                <span>Initializing Simulation</span>
                                                <span>Complete</span>
                                            </Typography>
                                            <LinearProgress variant="determinate" value={100} sx={{ height: 6, borderRadius: 3 }} />
                                        </Box>
                                        
                                        <Box sx={{ width: '100%' }}>
                                            <Typography variant="body2" fontWeight="500" sx={{ mb: 0.5, display: 'flex', justifyContent: 'space-between' }}>
                                                <span>Monte Carlo Sampling</span>
                                                <span>70%</span>
                                            </Typography>
                                            <LinearProgress variant="determinate" value={70} sx={{ height: 6, borderRadius: 3 }} />
                                        </Box>
                                        
                                        <Box sx={{ width: '100%' }}>
                                            <Typography variant="body2" fontWeight="500" sx={{ mb: 0.5, display: 'flex', justifyContent: 'space-between' }}>
                                                <span>Processing Results</span>
                                                <span>Pending</span>
                                            </Typography>
                                            <LinearProgress variant="determinate" value={0} sx={{ height: 6, borderRadius: 3 }} />
                                        </Box>
                                    </Stack>
                                </Box>
                                
                                <Box sx={{ mt: 4, px: 4, py: 2, bgcolor: 'background.paper', borderRadius: 2, boxShadow: 1 }}>
                                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', textAlign: 'center' }}>
                                        Processing {config.monte_carlo_samples} samples with {config.atmospheric_model.toUpperCase()} atmospheric model
                                        <br />
                                        Wind model: {config.wind_model.toUpperCase()} | Breakup modeling: {config.breakup_model.enabled ? 'Enabled' : 'Disabled'}
                                    </Typography>
                                </Box>
                            </Box>
                        ) : error ? (
                            <Box sx={{ p: 3 }}>
                                <Alert 
                                    severity="error" 
                                    variant="filled"
                                    sx={{ mb: 3 }}
                                    action={
                                        <IconButton 
                                            color="inherit" 
                                            size="small"
                                            onClick={() => setError(null)}
                                        >
                                            <DeleteOutline />
                                        </IconButton>
                                    }
                                >
                                    <AlertTitle>Analysis Error</AlertTitle>
                                    {error}
                                </Alert>
                                
                                <Card sx={{ mb: 3, borderLeft: '4px solid', borderColor: 'error.main' }}>
                                    <CardContent>
                                        <Typography variant="h6" gutterBottom>
                                            Troubleshooting Suggestions
                                        </Typography>
                                        <List dense>
                                            <ListItem>
                                                <ListItemIcon>
                                                    <CheckCircle fontSize="small" color="info" />
                                                </ListItemIcon>
                                                <ListItemText 
                                                    primary="Verify your initial state parameters" 
                                                    secondary="Ensure the position and velocity values are realistic"
                                                />
                                            </ListItem>
                                            <ListItem>
                                                <ListItemIcon>
                                                    <CheckCircle fontSize="small" color="info" />
                                                </ListItemIcon>
                                                <ListItemText 
                                                    primary="Reduce Monte Carlo sample count" 
                                                    secondary="High sample counts can cause timeouts or memory issues"
                                                />
                                            </ListItem>
                                            <ListItem>
                                                <ListItemIcon>
                                                    <CheckCircle fontSize="small" color="info" />
                                                </ListItemIcon>
                                                <ListItemText 
                                                    primary="Try a simpler atmospheric model" 
                                                    secondary="The exponential model is more reliable for quick tests"
                                                />
                                            </ListItem>
                                        </List>
                                    </CardContent>
                                </Card>
                                
                                <Button 
                                    variant="outlined" 
                                    startIcon={<Refresh />}
                                    onClick={() => setError(null)}
                                >
                                    Retry with Current Configuration
                                </Button>
                            </Box>
                        ) : trajectoryData ? (
                            <TrajectoryViewer 
                                trajectory={trajectoryData.trajectory} 
                                impactPrediction={trajectoryData.impactPrediction}
                                breakupPoints={trajectoryData.breakupPoints}
                            />
                        ) : (
                            // Use mock data for development
                            <TrajectoryViewer 
                                trajectory={mockData.trajectory} 
                                impactPrediction={mockData.impactPrediction}
                                breakupPoints={mockData.breakupPoints}
                            />
                        )}
                    </Paper>
                </Grid>
            </Grid>
            
            <Snackbar
                open={Boolean(successMessage)}
                autoHideDuration={4000}
                onClose={handleCloseSnackbar}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                sx={{ bottom: 24 }}
            >
                <Alert 
                    onClose={handleCloseSnackbar}
                    severity="success"
                    variant="filled"
                    sx={{ 
                        width: '100%',
                        boxShadow: 4,
                        '& .MuiAlert-icon': {
                            fontSize: '1.5rem'
                        }
                    }}
                    icon={<CheckCircle fontSize="inherit" />}
                >
                    <AlertTitle>Success</AlertTitle>
                    {successMessage}
                </Alert>
            </Snackbar>
        </Box>
    );
};

export default TrajectoryAnalysis;