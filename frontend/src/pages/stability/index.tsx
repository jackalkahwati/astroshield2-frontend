import React from 'react';
import Layout from '@/components/Layout';
import StabilityAnalysis from '@/components/stability/StabilityAnalysis';
import { Box } from '@mui/material';

const StabilityPage = () => {
  return (
    <Layout>
      <Box p={3}>
        <StabilityAnalysis />
      </Box>
    </Layout>
  );
};

export default StabilityPage;
