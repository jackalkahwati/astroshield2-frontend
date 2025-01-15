import axios from 'axios';

export interface DailyTrend {
  timestamp: string;
  totalScans: number;
  vulnerabilitiesFound: number;
  criticalIssues: number;
}

export interface AnalyticsData {
  dailyTrends: DailyTrend[];
  metrics: {
    totalScans: number;
    vulnerabilitiesFound: number;
    criticalIssues: number;
    averageResponseTime: number;
  };
}

// Mock data for development
const mockAnalyticsData: AnalyticsData = {
  dailyTrends: [
    {
      timestamp: '2024-01-14',
      totalScans: 150,
      vulnerabilitiesFound: 23,
      criticalIssues: 5
    },
    {
      timestamp: '2024-01-15',
      totalScans: 180,
      vulnerabilitiesFound: 28,
      criticalIssues: 7
    }
  ],
  metrics: {
    totalScans: 330,
    vulnerabilitiesFound: 51,
    criticalIssues: 12,
    averageResponseTime: 2.3
  }
};

export async function fetchAnalyticsData(): Promise<AnalyticsData> {
  try {
    // For development, return mock data
    return mockAnalyticsData;
    
    // For production, uncomment and use actual API call:
    // const response = await axios.get('/api/analytics');
    // return response.data;
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    throw new Error('Failed to fetch analytics data');
  }
} 