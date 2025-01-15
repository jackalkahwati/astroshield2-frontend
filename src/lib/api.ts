import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

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
    // For now, return mock data
    return {
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
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    throw new Error('Failed to fetch analytics data');
  }
} 