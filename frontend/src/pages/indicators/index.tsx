import React from 'react';
import { Box } from '@mui/material';
import Layout from '../../components/Layout';
import IndicatorSpecifications from '../../components/IndicatorSpecifications';

const IndicatorsPage: React.FC = () => {
  return (
    <Layout>
      <Box sx={{ p: 3 }}>
        <IndicatorSpecifications />
      </Box>
    </Layout>
  );
};

export default IndicatorsPage; 