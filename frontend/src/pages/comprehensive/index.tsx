import React, { useEffect, useState } from 'react';
import { Box } from '@mui/material';
import Layout from '../../components/Layout';
import ComprehensiveDashboard from '../../components/ComprehensiveDashboard';
import { API_CONFIG } from '../../lib/config';

interface ComprehensiveData {
  metrics: {
    attitude_stability: number;
    orbit_stability: number;
    thermal_stability: number;
    power_stability: number;
    communication_stability: number;
  };
  status: string;
  alerts: any[];
  timestamp: string;
}

const ComprehensivePage = () => {
  const [data, setData] = useState<ComprehensiveData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.comprehensive}`);
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        console.error('Error fetching data:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Layout>
      <Box p={3}>
        <ComprehensiveDashboard 
          data={data || undefined}
          isLoading={isLoading}
          error={error || undefined}
        />
      </Box>
    </Layout>
  );
};

export default ComprehensivePage;
