import React from 'react';
import { Box, Typography } from '@mui/material';
import Layout from '../../components/Layout';
import SettingsPanel from '../../components/settings/SettingsPanel';

const SettingsPage: React.FC = () => {
  return (
    <Layout>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
        <SettingsPanel />
      </Box>
    </Layout>
  );
};

export default SettingsPage; 