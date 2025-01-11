import React from 'react';
import { Box, Typography } from '@mui/material';
import Layout from '../../components/Layout';
import ManeuverPlanner from '../../components/maneuvers/ManeuverPlanner';

const ManeuversPage: React.FC = () => {
  return (
    <Layout>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Maneuver Planning
        </Typography>
        <ManeuverPlanner />
      </Box>
    </Layout>
  );
};

export default ManeuversPage; 