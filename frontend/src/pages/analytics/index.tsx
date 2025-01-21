import React from 'react';
import Layout from '@/components/Layout';
import AnalyticsDashboard from '@/components/analytics/AnalyticsDashboard';
import { Box } from '@mui/material';

const AnalyticsPage = () => {
  return (
    <Layout>
      <Box p={3}>
        <AnalyticsDashboard />
      </Box>
    </Layout>
  );
};

export default AnalyticsPage;
