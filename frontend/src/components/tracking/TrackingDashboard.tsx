import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Grid,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
} from '@mui/material';
import config from '@/config';

interface TrackingData {
  objectId: string;
  name: string;
  orbitType: string;
  altitude: number;
  velocity: number;
  lastUpdate: string;
  status: 'normal' | 'warning' | 'critical';
}

const mockTrackingData: TrackingData[] = [
  {
    objectId: 'SAT-001',
    name: 'StarLink-1234',
    orbitType: 'LEO',
    altitude: 550,
    velocity: 7.8,
    lastUpdate: '2024-03-27T10:30:00Z',
    status: 'normal',
  },
  {
    objectId: 'SAT-002',
    name: 'GPS-IIIA-06',
    orbitType: 'MEO',
    altitude: 20200,
    velocity: 3.9,
    lastUpdate: '2024-03-27T10:29:00Z',
    status: 'warning',
  },
];

const TrackingDashboard: React.FC = () => {
  const [trackingData, setTrackingData] = useState<TrackingData[]>(mockTrackingData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    const fetchTrackingData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // In development, use mock data with simulated delay
        if (process.env.NODE_ENV === 'development') {
          await new Promise(resolve => setTimeout(resolve, 1000));
          // Randomly modify some values to simulate real-time changes
          const updatedData = mockTrackingData.map(item => ({
            ...item,
            altitude: item.altitude + (Math.random() - 0.5) * 10,
            velocity: item.velocity + (Math.random() - 0.5) * 0.2,
            lastUpdate: new Date().toISOString(),
            status: Math.random() > 0.9 ? 'warning' : item.status,
          }));
          setTrackingData(updatedData);
        } else {
          const response = await fetch(`${config.apiUrl}/api/tracking/data`);
          if (!response.ok) {
            throw new Error('Failed to fetch tracking data');
          }
          const data = await response.json();
          setTrackingData(data);
        }
        
        setLastUpdate(new Date());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchTrackingData();

    // Set up polling interval
    const interval = setInterval(fetchTrackingData, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: TrackingData['status']) => {
    switch (status) {
      case 'warning':
        return '#ffa726';
      case 'critical':
        return '#f44336';
      default:
        return '#4caf50';
    }
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error: {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Last Update: {lastUpdate.toLocaleString()}
            </Typography>
            {loading && <CircularProgress size={20} />}
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Tracks
              </Typography>
              <Typography variant="h4">
                {trackingData.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alerts
              </Typography>
              <Typography variant="h4" sx={{ color: 'warning.main' }}>
                {trackingData.filter(d => d.status !== 'normal').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Update Rate
              </Typography>
              <Typography variant="h4">
                5s
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Tracking Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Active Tracking Data
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Object ID</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>Orbit Type</TableCell>
                    <TableCell>Altitude (km)</TableCell>
                    <TableCell>Velocity (km/s)</TableCell>
                    <TableCell>Last Update</TableCell>
                    <TableCell>Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trackingData.map((row) => (
                    <TableRow key={row.objectId}>
                      <TableCell>{row.objectId}</TableCell>
                      <TableCell>{row.name}</TableCell>
                      <TableCell>{row.orbitType}</TableCell>
                      <TableCell>{row.altitude.toFixed(2)}</TableCell>
                      <TableCell>{row.velocity.toFixed(2)}</TableCell>
                      <TableCell>{new Date(row.lastUpdate).toLocaleString()}</TableCell>
                      <TableCell>
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                          }}
                        >
                          <Box
                            sx={{
                              width: 10,
                              height: 10,
                              borderRadius: '50%',
                              backgroundColor: getStatusColor(row.status),
                            }}
                          />
                          {row.status}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TrackingDashboard;
