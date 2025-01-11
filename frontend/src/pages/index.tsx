import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { CircularProgress, Box, Typography } from '@mui/material';

export default function Home() {
  const router = useRouter();

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
