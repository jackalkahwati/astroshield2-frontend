import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Tabs,
  Tab,
  Paper,
  Button,
  IconButton,
  Tooltip,
  Chip,
  Stack,
  Avatar,
  Select,
  MenuItem,
  FormControl,
  LinearProgress,
} from '@mui/material';
import {
  Refresh,
  Save,
  PlayArrow,
  ArrowForwardIos,
  SatelliteAlt,
  WarningAmber,
  RocketLaunch,
  Public,
  Language,
  Visibility,
} from '@mui/icons-material';
import { styled, alpha } from '@mui/material/styles';
import TrajectoryViewer from '../components/TrajectoryViewer';

// --- Styled helpers --------------------------------------------------------
const HeaderRow = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing(3),
}));

const ScenarioBox = styled(Paper)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  cursor: 'pointer',
  transition: theme.transitions.create(['background', 'box-shadow'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    boxShadow: theme.shadows[4],
  },
}));

// --- Data types -----------------------------------------------------------
interface TrajectoryConfig {
  atmospheric_model: string;
  wind_model: string;
  monte_carlo_samples: number;
  object_properties: { mass: number; area: number; cd: number };
  breakup_model: { enabled: boolean; fragment_count: number; mass_distribution: string; velocity_perturbation: number };
}

// Quick-setup scenarios (re-using existing three) --------------------------
const scenarios = [
  {
    name: 'ISS Re-entry',
    description: 'International Space Station controlled de-orbit scenario',
    icon: <SatelliteAlt />,
    config: {
      atmospheric_model: 'nrlmsise',
      wind_model: 'hwm14',
      monte_carlo_samples: 1000,
      object_properties: { mass: 420000, area: 800, cd: 2.2 },
      breakup_model: { enabled: true, fragment_count: 50, mass_distribution: 'log_normal', velocity_perturbation: 25 },
    } as TrajectoryConfig,
    chips: ['NRLMSISE-00'],
  },
  {
    name: 'Tiangong Debris',
    description: 'Uncontrolled space station component',
    icon: <WarningAmber />,
    config: {
      atmospheric_model: 'jb2008',
      wind_model: 'hwm14',
      monte_carlo_samples: 1500,
      object_properties: { mass: 8000, area: 100, cd: 2.3 },
      breakup_model: { enabled: true, fragment_count: 12, mass_distribution: 'log_normal', velocity_perturbation: 45 },
    } as TrajectoryConfig,
    chips: ['JB2008', '12 fragments'],
  },
  {
    name: 'Falcon 9 Upper Stage',
    description: 'Spent rocket body in elliptical orbit',
    icon: <RocketLaunch />,
    config: {
      atmospheric_model: 'nrlmsise',
      wind_model: 'hwm14',
      monte_carlo_samples: 800,
      object_properties: { mass: 3500, area: 60, cd: 1.8 },
      breakup_model: { enabled: false, fragment_count: 0, mass_distribution: 'log_normal', velocity_perturbation: 0 },
    } as TrajectoryConfig,
    chips: ['NRLMSISE-00'],
  },
];

const defaultConfig = scenarios[0].config;

// Mock comparison dataset – three model runs (ground-truth, physics, ML)
const mockTrajectoryData = {
  models: [
    {
      name: 'Ground Truth',
      color: '#4CAF50',
      trajectory: Array.from({ length: 100 }, (_, i) => ({
        time: i * 10,
        position: [-74.006 + i * 0.010, 40.7128 + i * 0.010, 100000 - i * 1000] as [number, number, number],
        velocity: [1000 - i * 10, 0, -1000 - i * 10] as [number, number, number],
      })),
    },
    {
      name: 'Physics Propagation',
      color: '#1E90FF',
      trajectory: Array.from({ length: 100 }, (_, i) => ({
        time: i * 10,
        position: [-74.016 + i * 0.012, 40.7028 + i * 0.011, 100050 - i * 1005] as [number, number, number],
        velocity: [995 - i * 10, 0, -1003 - i * 9] as [number, number, number],
      })),
    },
    {
      name: 'ML Forecast',
      color: '#FF5722',
      trajectory: Array.from({ length: 100 }, (_, i) => ({
        time: i * 10,
        position: [-73.996 + i * 0.009, 40.7228 + i * 0.012, 99900 - i * 995] as [number, number, number],
        velocity: [998 - i * 9, 0, -1002 - i * 11] as [number, number, number],
      })),
    },
  ],
  impactPrediction: {
    time: new Date().toISOString(),
    location: { lat: 40.7128, lon: -74.006 },
    velocity: { magnitude: 900, direction: { x: 0.7, y: 0, z: -0.7 } },
    uncertainty_radius_km: 10,
    confidence: 0.95,
  },
  breakupPoints: [
    { altitude: 80000, fragments: 5, time: new Date().toISOString() },
  ],
};

// -------------------------------------------------------------------------
export default function TrajectoryAnalysisV2() {
  const [tab, setTab] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [config, setConfig] = useState<TrajectoryConfig>(defaultConfig);
  const [loading, setLoading] = useState(false);
  const [trajectoryData, setTrajectoryData] = useState<any>(null);

  // Helper to create slightly different trajectories for each scenario so
  // the user sees unique lines instead of identical ones
  const buildMockDataForScenario = (index: number) => {
    // Offsets per scenario to make paths visibly different
    const offsets = [
      { dLon: 0.01, dLat: 0.01, label: 'ISS Re-entry' },
      { dLon: 0.015, dLat: -0.008, label: 'Tiangong Debris' },
      { dLon: -0.012, dLat: 0.013, label: 'Falcon 9 Stage' },
    ];
    const { dLon, dLat } = offsets[index] ?? offsets[0];

    const makeTraj = (mult = 1) =>
      Array.from({ length: 100 }, (_, i) => ({
        time: i * 10,
        position: [
          -74.006 + i * dLon * mult,
          40.7128 + i * dLat * mult,
          100000 - i * 1000,
        ] as [number, number, number],
        velocity: [1000 - i * 10, 0, -1000 - i * 10] as [number, number, number],
      }));

    return {
      models: [
        { name: 'Ground Truth', color: '#4CAF50', trajectory: makeTraj(1) },
        { name: 'Physics Propagation', color: '#1E90FF', trajectory: makeTraj(1.1) },
        { name: 'ML Forecast', color: '#FF5722', trajectory: makeTraj(0.9) },
      ],
      impactPrediction: {
        time: new Date().toISOString(),
        location: { lat: 40.7128 + 25 * dLat, lon: -74.006 + 25 * dLon },
        velocity: { magnitude: 900, direction: { x: 0.7, y: 0, z: -0.7 } },
        uncertainty_radius_km: 10,
        confidence: 0.95,
      },
      breakupPoints: [
        { altitude: 80000, fragments: 5, time: new Date().toISOString() },
      ],
    };
  };

  const runAnalysis = async () => {
    if (selectedIndex === null) return;
    setLoading(true);
    // TODO: call backend; for now simulate delay
    await new Promise((r) => setTimeout(r, 800));
    const data = buildMockDataForScenario(selectedIndex);
    setTrajectoryData(data);
    setLoading(false);
  };

  const resetPage = () => {
    setSelectedIndex(null);
    setTrajectoryData(null);
  };

  // ---- Render helpers ----------------------------------------------------
  const readyState = !trajectoryData && !loading;

  return (
    <Box>
      {/* Header row -------------------------------------------------------*/}
      <HeaderRow>
        <Box>
          <Typography variant="h4" fontWeight={600}>Trajectory Analysis</Typography>
          <Typography variant="body2" color="text.secondary" maxWidth={600}>
            Analyze spacecraft trajectories, predict re-entry paths, and assess impact risks.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={resetPage}>
            Reset
          </Button>
          <Button variant="outlined" startIcon={<Save />} color="secondary">
            Save
          </Button>
        </Stack>
      </HeaderRow>

      {/* Main two-column grid --------------------------------------------*/}
      <Grid container spacing={3}>
        {/* LEFT : configuration */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, bgcolor: (t) => alpha(t.palette.background.paper, 0.05) }}> {/* subtle tint */}
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              variant="fullWidth"
              sx={{ mb: 2 }}
            >
              <Tab label="Quick Setup" />
              <Tab label="Advanced" />
              <Tab label="History" />
            </Tabs>

            {tab === 0 && (
              <Box>
                <Typography variant="subtitle1" mb={1} fontWeight={600}>Select Scenario</Typography>
                <Stack spacing={1} mb={2}>
                  {scenarios.map((s, idx) => (
                    <ScenarioBox key={s.name} onClick={() => { setSelectedIndex(idx); setConfig(s.config); }}
                      sx={{
                        bgcolor: selectedIndex === idx ? (t) => alpha(t.palette.primary.main, 0.15) : undefined,
                      }}
                    >
                      <Stack direction="row" alignItems="center" spacing={2}>
                        <Avatar>{s.icon}</Avatar>
                        <Box>
                          <Typography fontWeight={600}>{s.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{s.description}</Typography>
                          <Stack direction="row" spacing={1} mt={0.5}>
                            {s.chips.map((c) => <Chip key={c} label={c} size="small" />)}
                          </Stack>
                        </Box>
                      </Stack>
                      <ArrowForwardIos fontSize="small" />
                    </ScenarioBox>
                  ))}
                </Stack>

                <Button fullWidth variant="contained" startIcon={<PlayArrow />} onClick={runAnalysis} disabled={loading || selectedIndex === null}>
                  {loading ? 'Running…' : 'Run Trajectory Analysis'}
                </Button>
              </Box>
            )}

            {tab === 1 && (
              <Box>
                <Typography variant="body2">Advanced parameters coming soon…</Typography>
              </Box>
            )}

            {tab === 2 && (
              <Box>
                <Typography variant="body2">No previous analyses.</Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* RIGHT : visualization */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6" fontWeight={600}>Trajectory Visualization</Typography>
              <Stack direction="row" spacing={1}>
                <Button size="small" startIcon={<Refresh />} onClick={() => {/* TODO reset view */ }}>
                  Reset View
                </Button>
                <FormControl size="small">
                  <Select value="3d" onChange={() => { }}>
                    <MenuItem value="3d">3D Globe</MenuItem>
                    <MenuItem value="2d">2D Map</MenuItem>
                  </Select>
                </FormControl>
              </Stack>
            </Box>

            {/* Content area ----------------------------------------------*/}
            {loading && (
              <Box textAlign="center" py={6}>
                <LinearProgress />
                <Typography variant="body2" mt={2}>Running analysis…</Typography>
              </Box>
            )}

            {readyState && !loading && (
              <Box height="400px" display="flex" flexDirection="column" alignItems="center" justifyContent="center">
                <Language sx={{ fontSize: 64, color: 'primary.main' }} />
                <Typography variant="h6" fontWeight={600} mt={2}>Ready for Analysis</Typography>
                <Typography variant="body2" color="text.secondary" mb={3} textAlign="center" maxWidth={300}>
                  Select a scenario and run the analysis to visualize trajectory results
                </Typography>
                <Button variant="contained" startIcon={<PlayArrow />} onClick={runAnalysis} disabled={selectedIndex === null}>Start Analysis</Button>
              </Box>
            )}

            {trajectoryData && !loading && (
              <TrajectoryViewer
                models={trajectoryData.models}
                impactPrediction={trajectoryData.impactPrediction}
                breakupPoints={trajectoryData.breakupPoints}
              />
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 