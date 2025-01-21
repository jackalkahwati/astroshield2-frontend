import React from 'react';
import Layout from '@/components/Layout';
import ManeuverPlanner from '@/components/maneuvers/ManeuverPlanner';
import { Box } from '@mui/material';

const ManeuversPage = () => {
  return (
    <Layout>
      <Box p={3}>
        <ManeuverPlanner />
      </Box>
    </Layout>
  );
};

export default ManeuversPage;
