import axios from 'axios';

// Base API URL, can be configured from environment variables
const API_URL = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

// Create axios instance with common configuration
const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add request interceptor for authentication
api.interceptors.request.use((config) => {
  // Get token from localStorage or other state management
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Add response interceptor for error handling
api.interceptors.response.use((response) => {
  return response;
}, (error) => {
  // Handle common errors
  if (error.response) {
    // Server responded with error
    if (error.response.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
  } else if (error.request) {
    // Request made but no response received
    console.error('Network error:', error.request);
  } else {
    // Error setting up request
    console.error('Request error:', error.message);
  }
  return Promise.reject(error);
});

// CCDM API functions
const ccdmService = {
  // Analyze a space object
  analyzeObject: async (noradId, data = {}) => {
    try {
      const response = await api.post('/ccdm/analyze', {
        norad_id: noradId,
        data
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing object:', error);
      throw error;
    }
  },

  // Assess threat level for a space object
  assessThreat: async (noradId, data = {}) => {
    try {
      const response = await api.post('/ccdm/threat-assessment', {
        norad_id: noradId,
        data
      });
      return response.data;
    } catch (error) {
      console.error('Error assessing threat:', error);
      throw error;
    }
  },

  // Quick threat assessment using just NORAD ID
  quickAssess: async (noradId) => {
    try {
      const response = await api.get(`/ccdm/quick-assessment/${noradId}`);
      return response.data;
    } catch (error) {
      console.error('Error performing quick assessment:', error);
      throw error;
    }
  },

  // Get historical analysis for a date range
  getHistoricalAnalysis: async (noradId, startDate, endDate) => {
    try {
      const response = await api.post('/ccdm/historical', {
        norad_id: noradId,
        start_date: startDate,
        end_date: endDate
      });
      return response.data;
    } catch (error) {
      console.error('Error getting historical analysis:', error);
      throw error;
    }
  },

  // Get last week's analysis (convenience method)
  getLastWeekAnalysis: async (noradId) => {
    try {
      const response = await api.get(`/ccdm/last-week-analysis/${noradId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting last week analysis:', error);
      throw error;
    }
  },

  // Detect shape changes for a space object
  detectShapeChanges: async (noradId, startDate, endDate, data = {}) => {
    try {
      const response = await api.post('/ccdm/shape-changes', {
        norad_id: noradId,
        start_date: startDate,
        end_date: endDate,
        data
      });
      return response.data;
    } catch (error) {
      console.error('Error detecting shape changes:', error);
      throw error;
    }
  }
};

export default ccdmService; 