import React from 'react';
import { Box, Typography } from '@mui/material';
import Layout from '../../components/Layout';
import StabilityAnalysis from '../../components/stability/StabilityAnalysis';

const StabilityPage: React.FC = () => {
  return (
    <Layout>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Stability Analysis
        </Typography>
        <StabilityAnalysis />
      </Box>
    </Layout>
  );
};

export default StabilityPage; 