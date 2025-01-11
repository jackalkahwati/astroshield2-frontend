import React from 'react';
import { useRouter } from 'next/router';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Satellite as SatelliteIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  List as ListIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

const Navigation: React.FC = () => {
  const router = useRouter();
  const currentPath = router.pathname || '/';

  const menuItems = [
    { text: 'Comprehensive', icon: <AssessmentIcon />, path: '/comprehensive' },
    { text: 'Indicators', icon: <ListIcon />, path: '/indicators' },
    { text: 'Satellite Tracking', icon: <SatelliteIcon />, path: '/tracking' },
    { text: 'Stability Analysis', icon: <AssessmentIcon />, path: '/stability' },
    { text: 'Maneuvers', icon: <SecurityIcon />, path: '/maneuvers' },
    { text: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
  ];

  const handleNavigation = async (path: string) => {
    if (!router.isReady || !path) return;
    try {
      await router.push(path);
    } catch (error) {
      console.error('Navigation error:', error);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={currentPath === item.path}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                },
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ListItemIcon sx={{ color: currentPath === item.path ? 'white' : 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => handleNavigation('/settings')}
            selected={currentPath === '/settings'}
            sx={{
              '&.Mui-selected': {
                backgroundColor: 'primary.main',
                color: 'white',
                '&:hover': {
                  backgroundColor: 'primary.dark',
                },
              },
              '&:hover': {
                backgroundColor: 'action.hover',
              },
            }}
          >
            <ListItemIcon sx={{ color: currentPath === '/settings' ? 'white' : 'inherit' }}>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );
};

export default Navigation;
