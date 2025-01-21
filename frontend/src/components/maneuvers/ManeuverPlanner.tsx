import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Snackbar,
  CircularProgress,
  SelectChangeEvent,
} from '@mui/material';
import api from '@/services/api';
import config from '@/config';

interface Maneuver {
  id: string;
  type: string;
  status: string;
  scheduledTime: string;
  completedTime?: string;
  details: {
    deltaV?: number;
    duration?: number;
    fuel_required?: number;
    rotation_angle?: number;
    fuel_used?: number;
  };
}

interface ManeuverData {
  maneuvers: Maneuver[];
  resources: {
    fuel_remaining: number;
    thrust_capacity: number;
    next_maintenance: string;
  };
  lastUpdate: string;
}

interface ManeuverFormData {
  type: string;
  deltaV: number;
  executionTime: string;
}

const ManeuverPlanner: React.FC = () => {
  const [maneuverData, setManeuverData] = useState<ManeuverData | null>(null);
  const [formData, setFormData] = useState<ManeuverFormData>({
    type: '',
    deltaV: 0,
    executionTime: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    const loadManeuvers = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/maneuvers/data');
        if (!response.ok) {
          throw new Error('Failed to fetch maneuvers data');
        }
        const data = await response.json();
        setManeuverData(data);
        setError(null);
      } catch (err) {
        console.error('Error loading maneuvers:', err);
        setError('Failed to load maneuvers data');
        setManeuverData(null);
      } finally {
        setLoading(false);
      }
    };

    loadManeuvers();
    const interval = setInterval(loadManeuvers, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleSelectChange = (event: SelectChangeEvent) => {
    setFormData(prev => ({
      ...prev,
      type: event.target.value,
    }));
  };

  const handleNumberChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(event.target.value);
    setFormData(prev => ({
      ...prev,
      deltaV: isNaN(value) ? 0 : value,
    }));
  };

  const handleDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      executionTime: event.target.value,
    }));
  };

  const validateForm = () => {
    if (!formData.type) {
      setError('Please select a maneuver type');
      return false;
    }
    if (!formData.deltaV || formData.deltaV <= 0) {
      setError('Please enter a valid Delta-V value');
      return false;
    }
    if (!formData.executionTime) {
      setError('Please select an execution time');
      return false;
    }
    if (new Date(formData.executionTime) < new Date()) {
      setError('Execution time must be in the future');
      return false;
    }
    return true;
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(null);

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/maneuvers/plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to plan maneuver');
      }

      const newManeuver = await response.json();
      if (maneuverData) {
        setManeuverData({
          ...maneuverData,
          maneuvers: [newManeuver, ...maneuverData.maneuvers]
        });
      }
      setSuccess('Maneuver planned successfully');
      
      // Reset form
      setFormData({
        type: '',
        deltaV: 0,
        executionTime: '',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to plan maneuver');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled':
        return '#2196f3';
      case 'completed':
        return '#4caf50';
      case 'failed':
        return '#f44336';
      case 'executing':
        return '#ffa726';
      default:
        return '#2196f3';
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Maneuver Planning Form */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Plan New Maneuver
            </Typography>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Maneuver Type</InputLabel>
                    <Select
                      value={formData.type}
                      onChange={handleSelectChange}
                      label="Maneuver Type"
                    >
                      <MenuItem value="hohmann">Hohmann Transfer</MenuItem>
                      <MenuItem value="stationkeeping">Station Keeping</MenuItem>
                      <MenuItem value="phasing">Phasing Maneuver</MenuItem>
                      <MenuItem value="collision">Collision Avoidance</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Delta-V (km/s)"
                    type="number"
                    value={formData.deltaV}
                    onChange={handleNumberChange}
                    inputProps={{ step: 0.1, min: 0 }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Execution Time"
                    type="datetime-local"
                    value={formData.executionTime}
                    onChange={handleDateChange}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    type="submit"
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
                  >
                    {loading ? 'Planning...' : 'Plan Maneuver'}
                  </Button>
                </Grid>
              </Grid>
            </form>
          </Paper>
        </Grid>

        {/* Active Maneuvers List */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Maneuvers
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Delta-V (m/s)</TableCell>
                    <TableCell>Execution Time</TableCell>
                    <TableCell>Duration (s)</TableCell>
                    <TableCell>Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <CircularProgress />
                      </TableCell>
                    </TableRow>
                  ) : error ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Alert severity="error">{error}</Alert>
                      </TableCell>
                    </TableRow>
                  ) : !maneuverData || maneuverData.maneuvers.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography color="textSecondary">
                          No maneuvers planned
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    maneuverData.maneuvers.map((maneuver) => (
                      <TableRow key={maneuver.id}>
                        <TableCell>{maneuver.id}</TableCell>
                        <TableCell>{maneuver.type}</TableCell>
                        <TableCell>{maneuver.details.deltaV?.toFixed(2) || 'N/A'}</TableCell>
                        <TableCell>
                          {new Date(maneuver.scheduledTime).toLocaleString()}
                        </TableCell>
                        <TableCell>{maneuver.details.duration || 'N/A'}</TableCell>
                        <TableCell>
                          <Typography
                            sx={{
                              color: getStatusColor(maneuver.status),
                              fontWeight: "bold"
                            }}
                          >
                            {maneuver.status.toUpperCase()}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Error/Success Messages */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ManeuverPlanner;
