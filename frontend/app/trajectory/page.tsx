'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { Box, Typography } from '@mui/material';

// Dynamically import the TrajectoryAnalysis component with no SSR
// This is necessary because mapbox-gl is a client-side library
const TrajectoryAnalysis = dynamic(
  () => import('@/src/containers/TrajectoryAnalysisV2'),
  { ssr: false }
);

export default function TrajectoryPage() {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Trajectory Analysis
      </Typography>
      <Typography variant="body1" paragraph>
        Analyze spacecraft trajectories, predict reentry paths, and assess impact risks.
      </Typography>
      
      <TrajectoryAnalysis />
    </Box>
  );
} 