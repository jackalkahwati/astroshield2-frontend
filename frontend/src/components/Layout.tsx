import React, { ReactNode } from 'react';
import { Box, Container, Typography } from '@mui/material';
import Navigation from './Navigation';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <Box
        component="nav"
        sx={{
          width: { xs: 0, sm: 240 },
          flexShrink: 0,
          display: { xs: 'none', sm: 'block' },
          borderRight: '1px solid',
          borderColor: 'divider',
          height: '100vh',
          position: 'fixed',
          bgcolor: 'background.paper',
          overflowY: 'auto',
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" component="div" noWrap>
            AstroShield
          </Typography>
        </Box>
        <Navigation />
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - 240px)` },
          ml: { sm: '240px' },
          minHeight: '100vh',
        }}
      >
        <Container maxWidth="xl">
          {children}
        </Container>
      </Box>
    </Box>
  );
};

export default Layout; 