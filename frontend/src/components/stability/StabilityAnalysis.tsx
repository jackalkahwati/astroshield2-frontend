import React, { useEffect, useState } from 'react';
import { Box, Grid, Paper, Typography, Card, CardContent, CircularProgress } from '@mui/material';
import { API_CONFIG } from '../../lib/config';

interface StabilityData {
  metrics: {
    attitude_stability: number;
    orbit_stability: number;
    thermal_stability: number;
    power_stability: number;
    communication_stability: number;
  };
  status: string;
  timestamp: string;
}

const StabilityAnalysis: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<StabilityData | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.stability}`);
        if (!response.ok) {
          throw new Error('Failed to fetch stability data');
        }
        const stabilityData = await response.json();
        setData(stabilityData);
        setError(null);
      } catch (err) {
        setError('Failed to load stability data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (value: number) => {
    if (value > 90) return 'success.main';
    if (value > 70) return 'warning.main';
    return 'error.main';
  };

  if (loading) {
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
    return null;
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Overall Status */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Stability Overview
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Typography variant="h5" sx={{ mr: 2 }}>
                Overall Status:
              </Typography>
              <Typography
                variant="h5"
                sx={{ color: data.status === 'nominal' ? 'success.main' : 'warning.main' }}
              >
                {data.status.toUpperCase()}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Last Updated: {new Date(data.timestamp).toLocaleString()}
            </Typography>
          </Paper>
        </Grid>

        {/* Stability Metrics */}
        {Object.entries(data.metrics).map(([key, value]) => (
          <Grid item xs={12} md={6} key={key}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h4" sx={{ color: getStatusColor(value) }}>
                    {value.toFixed(1)}%
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="body1" color={getStatusColor(value)}>
                    Status: {value > 90 ? 'STABLE' : value > 70 ? 'WARNING' : 'CRITICAL'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default StabilityAnalysis;