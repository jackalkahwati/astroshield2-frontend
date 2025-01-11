import React from 'react';
import { Box, Typography } from '@mui/material';
import Layout from '../../components/Layout';
import TrackingDashboard from '../../components/tracking/TrackingDashboard';

const TrackingPage: React.FC = () => {
  return (
    <Layout>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Satellite Tracking
        </Typography>
        <TrackingDashboard />
      </Box>
    </Layout>
  );
};

export default TrackingPage; 