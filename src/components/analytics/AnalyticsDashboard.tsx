import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { fetchAnalyticsData, AnalyticsData } from '@/lib/api';

interface DailyTrend {
  timestamp: string;
  totalScans: number;
  vulnerabilitiesFound: number;
  criticalIssues: number;
}

export default function AnalyticsDashboard() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const analyticsData = await fetchAnalyticsData();
        setData(analyticsData);
        setError(null);
      } catch (err) {
        setError('Failed to load analytics data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>No data available</div>;

  return (
    <div className="w-full h-96 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data.dailyTrends}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="totalScans" stroke="#8884d8" />
          <Line type="monotone" dataKey="vulnerabilitiesFound" stroke="#82ca9d" />
          <Line type="monotone" dataKey="criticalIssues" stroke="#ff7300" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
