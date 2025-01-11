import React from 'react';
import { Box, Grid, Paper, Typography, CircularProgress, Alert } from '@mui/material';

interface ComprehensiveData {
  metrics: {
    attitude_stability: number;
    orbit_stability: number;
    thermal_stability: number;
    power_stability: number;
    communication_stability: number;
  };
  status: string;
  alerts: any[];
  timestamp: string;
}

interface Props {
  data?: ComprehensiveData;
  isLoading: boolean;
  error?: string;
}

const ComprehensiveDashboard: React.FC<Props> = ({ data, isLoading, error }) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={2}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!data) {
    return (
      <Box p={2}>
        <Alert severity="info">No data available</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Comprehensive System Status
      </Typography>
      <Grid container spacing={3}>
        {Object.entries(data.metrics).map(([key, value]) => (
          <Grid item xs={12} sm={6} md={4} key={key}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              </Typography>
              <Typography variant="h4" color={value > 90 ? 'success.main' : value > 70 ? 'warning.main' : 'error.main'}>
                {value.toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>
      <Box mt={3}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            System Status: {data.status.toUpperCase()}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Last Updated: {new Date(data.timestamp).toLocaleString()}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
};

export default ComprehensiveDashboard; 