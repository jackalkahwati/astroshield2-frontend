import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { CircularProgress, Box, Typography } from '@mui/material';
import type { GetServerSideProps, InferGetServerSidePropsType } from 'next';

interface HomeProps {
  isHealthCheck: boolean;
}

export const getServerSideProps: GetServerSideProps<HomeProps> = async (context) => {
  // Set CORS headers
  context.res.setHeader('Access-Control-Allow-Origin', '*');
  context.res.setHeader('Access-Control-Allow-Methods', 'GET');
  
  // If it's a health check (from Railway or other monitoring)
  const userAgent = context.req.headers['user-agent']?.toLowerCase() || '';
  if (userAgent.includes('railway') || context.req.headers['x-health-check']) {
    context.res.statusCode = 200;
    return {
      props: {
        isHealthCheck: true
      }
    };
  }

  return {
    props: {
      isHealthCheck: false
    }
  };
};

export default function Home({ isHealthCheck }: HomeProps) {
  const router = useRouter();

  useEffect(() => {
    if (!isHealthCheck) {
      router.push('/comprehensive');
    }
  }, [router, isHealthCheck]);

  // Return a simple message for health checks
  if (isHealthCheck) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
      >
        <Typography variant="h6">OK</Typography>
      </Box>
    );
  }

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight="100vh"
      gap={2}
    >
      <CircularProgress size={40} />
      <Typography variant="h6">Loading AstroShield...</Typography>
    </Box>
  );
}
