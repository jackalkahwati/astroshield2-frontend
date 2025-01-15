import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { CircularProgress, Box, Typography } from '@mui/material';
import type { GetServerSideProps } from 'next';

interface HomeProps {
  isHealthCheck: boolean;
}

export const getServerSideProps: GetServerSideProps<HomeProps> = async ({ req }) => {
  // If it's a health check from Railway
  const userAgent = req.headers['user-agent']?.toLowerCase() || '';
  if (userAgent.includes('railway')) {
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

  // For health checks, return immediately
  if (isHealthCheck) {
    return null;
  }

  useEffect(() => {
    router.push('/comprehensive');
  }, [router]);

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
