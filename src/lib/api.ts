import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://astroshield2-api.vercel.app';

export const fetchAnalyticsData = async (timeRange: string = '24h') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/analytics/data`, {
      params: { timeRange }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    throw error;
  }
}; 