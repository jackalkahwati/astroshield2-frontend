import React from 'react';
import { Box, Typography, Grid, Paper, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

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

interface ComprehensiveDashboardProps {
  data?: ComprehensiveData;
  isLoading: boolean;
  error?: string;
}

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const MetricBox = ({ label, value }: { label: string; value: number }) => (
  <Box mb={2}>
    <Typography variant="subtitle2" color="textSecondary">
      {label}
    </Typography>
    <Typography variant="h6">
      {value.toFixed(1)}%
    </Typography>
  </Box>
);

const ComprehensiveDashboard: React.FC<ComprehensiveDashboardProps> = ({
  data,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (!data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography>No data available</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Status: {data.status}
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <Typography variant="h6" gutterBottom>
              Stability Metrics
            </Typography>
            <MetricBox label="Attitude Stability" value={data.metrics.attitude_stability} />
            <MetricBox label="Orbit Stability" value={data.metrics.orbit_stability} />
            <MetricBox label="Thermal Stability" value={data.metrics.thermal_stability} />
            <MetricBox label="Power Stability" value={data.metrics.power_stability} />
            <MetricBox label="Communication Stability" value={data.metrics.communication_stability} />
          </StyledPaper>
        </Grid>
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <Typography variant="h6" gutterBottom>
              Alerts
            </Typography>
            {data.alerts.length === 0 ? (
              <Typography>No active alerts</Typography>
            ) : (
              data.alerts.map((alert, index) => (
                <Box key={index} mb={1}>
                  <Typography>{alert}</Typography>
                </Box>
              ))
            )}
          </StyledPaper>
        </Grid>
      </Grid>
      <Box mt={2}>
        <Typography variant="caption" color="textSecondary">
          Last updated: {new Date(data.timestamp).toLocaleString()}
        </Typography>
      </Box>
    </Box>
  );
};

export default ComprehensiveDashboard;
