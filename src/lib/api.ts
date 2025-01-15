import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://astroshield2-api-production.up.railway.app';
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export interface AnalyticsData {
  dailyTrends: Array<{
    timestamp: string;
    totalScans: number;
    vulnerabilitiesFound: number;
    criticalIssues: number;
  }>;
  metrics: {
    totalScans: number;
    vulnerabilitiesFound: number;
    criticalIssues: number;
    averageResponseTime: number;
  };
}

export async function fetchAnalyticsData(): Promise<AnalyticsData> {
  try {
    const response = await api.get('/analytics');
    return response.data;
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    // Fallback mock data for development and error cases
    return {
      dailyTrends: [
        {
          timestamp: '2024-01-14',
          totalScans: 150,
          vulnerabilitiesFound: 23,
          criticalIssues: 5,
        },
        {
          timestamp: '2024-01-15',
          totalScans: 180,
          vulnerabilitiesFound: 28,
          criticalIssues: 7,
        },
      ],
      metrics: {
        totalScans: 330,
        vulnerabilitiesFound: 51,
        criticalIssues: 12,
        averageResponseTime: 2.3,
      },
    };
  }
} 