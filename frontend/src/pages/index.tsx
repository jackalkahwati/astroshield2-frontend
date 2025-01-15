import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { CircularProgress, Box, Typography } from '@mui/material';

// Add a getServerSideProps function to handle health checks
export async function getServerSideProps({ req, res }) {
  // If it's a health check (typically from Railway)
  if (req.headers['user-agent']?.toLowerCase().includes('railway')) {
    res.statusCode = 200;
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
}

export default function Home({ isHealthCheck }) {
  const router = useRouter();

  useEffect(() => {
    if (!isHealthCheck) {
      router.push('/comprehensive');
    }
  }, [router, isHealthCheck]);

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
