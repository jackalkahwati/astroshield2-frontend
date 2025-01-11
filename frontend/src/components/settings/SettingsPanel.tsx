import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Alert,
  Snackbar,
} from '@mui/material';
import { API_CONFIG } from '../../lib/config';

interface Settings {
  darkMode: boolean;
  updateInterval: number;
  notificationsEnabled: boolean;
  dataRetentionDays: number;
  displayUnit: 'metric' | 'imperial';
  language: string;
}

const defaultSettings: Settings = {
  darkMode: true,
  updateInterval: 5,
  notificationsEnabled: true,
  dataRetentionDays: 30,
  displayUnit: 'metric',
  language: 'en',
};

const SETTINGS_STORAGE_KEY = 'astroShieldSettings';

const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load settings from localStorage on component mount
    const savedSettings = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (savedSettings) {
      try {
        const parsedSettings = JSON.parse(savedSettings);
        setSettings(parsedSettings);
      } catch (err) {
        console.error('Failed to parse saved settings:', err);
        setError('Failed to load saved settings');
      }
    }
  }, []);

  const handleSwitchChange = (field: keyof Pick<Settings, 'darkMode' | 'notificationsEnabled'>) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSettings(prev => ({
      ...prev,
      [field]: event.target.checked,
    }));
  };

  const handleNumberChange = (field: keyof Pick<Settings, 'updateInterval' | 'dataRetentionDays'>) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value)) {
      setSettings(prev => ({
        ...prev,
        [field]: value,
      }));
    }
  };

  const handleSelectChange = (field: keyof Pick<Settings, 'displayUnit' | 'language'>) => (
    event: SelectChangeEvent<string>
  ) => {
    setSettings(prev => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const validateSettings = () => {
    if (settings.updateInterval < 1 || settings.updateInterval > 60) {
      setError('Update interval must be between 1 and 60 seconds');
      return false;
    }
    if (settings.dataRetentionDays < 1 || settings.dataRetentionDays > 365) {
      setError('Data retention must be between 1 and 365 days');
      return false;
    }
    return true;
  };

  const handleSave = async () => {
    try {
      setError(null);
      setSuccess(null);

      if (!validateSettings()) {
        return;
      }

      // Save to localStorage
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));

      // In production, you would also save to backend
      if (process.env.NODE_ENV !== 'development') {
        const response = await fetch(`${API_CONFIG.baseUrl}/api/settings`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(settings),
        });

        if (!response.ok) {
          throw new Error('Failed to save settings to server');
        }
      }

      setSuccess('Settings saved successfully');

      // Apply settings
      document.documentElement.setAttribute('data-theme', settings.darkMode ? 'dark' : 'light');
      // Add other settings applications as needed
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Display Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Display Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.darkMode}
                      onChange={handleSwitchChange('darkMode')}
                    />
                  }
                  label="Dark Mode"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel id="display-units-label">Display Units</InputLabel>
                  <Select
                    labelId="display-units-label"
                    value={settings.displayUnit}
                    onChange={handleSelectChange('displayUnit')}
                    label="Display Units"
                  >
                    <MenuItem value="metric">Metric</MenuItem>
                    <MenuItem value="imperial">Imperial</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel id="language-label">Language</InputLabel>
                  <Select
                    labelId="language-label"
                    value={settings.language}
                    onChange={handleSelectChange('language')}
                    label="Language"
                  >
                    <MenuItem value="en">English</MenuItem>
                    <MenuItem value="es">Spanish</MenuItem>
                    <MenuItem value="fr">French</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Data Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Data Settings
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Update Interval (seconds)"
                  value={settings.updateInterval}
                  onChange={handleNumberChange('updateInterval')}
                  inputProps={{ min: 1, max: 60 }}
                  error={settings.updateInterval < 1 || settings.updateInterval > 60}
                  helperText={settings.updateInterval < 1 || settings.updateInterval > 60 ? 
                    'Must be between 1 and 60 seconds' : ''}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Data Retention (days)"
                  value={settings.dataRetentionDays}
                  onChange={handleNumberChange('dataRetentionDays')}
                  inputProps={{ min: 1, max: 365 }}
                  error={settings.dataRetentionDays < 1 || settings.dataRetentionDays > 365}
                  helperText={settings.dataRetentionDays < 1 || settings.dataRetentionDays > 365 ? 
                    'Must be between 1 and 365 days' : ''}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Notification Settings
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.notificationsEnabled}
                  onChange={handleSwitchChange('notificationsEnabled')}
                />
              }
              label="Enable Notifications"
            />
          </Paper>
        </Grid>

        {/* Save Button */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSave}
              size="large"
            >
              Save Settings
            </Button>
          </Box>
        </Grid>
      </Grid>

      {/* Feedback Messages */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setSuccess(null)} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SettingsPanel; 