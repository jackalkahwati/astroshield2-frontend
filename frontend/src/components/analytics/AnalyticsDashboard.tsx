import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  SelectChangeEvent,
  Skeleton,
  IconButton,
  Tooltip as MuiTooltip,
} from '@mui/material';
import {
  Refresh,
  ZoomIn,
  ZoomOut,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';
import { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';

interface DailyTrend {
  timestamp: string;
  conjunction_count: number;
  threat_level: number;
  protection_coverage: number;
  response_time: number;
  mitigation_success: number;
}

interface AnalyticsData {
  summary: {
    total_conjunctions_analyzed: number;
    threats_detected: number;
    threats_mitigated: number;
    average_response_time: number;
    protection_coverage: number;
  };
  current_metrics: {
    threat_analysis: {
      value: number;
      trend: string;
      status: string;
    };
    collision_avoidance: {
      value: number;
      trend: string;
      status: string;
    };
    debris_tracking: {
      value: number;
      trend: string;
      status: string;
    };
    protection_status: {
      value: number;
      trend: string;
      status: string;
    };
  };
  trends: {
    daily: DailyTrend[];
    weekly_summary: {
      average_threat_level: number;
      total_conjunctions: number;
      mitigation_success_rate: number;
      average_response_time: number;
    };
    monthly_summary: {
      average_threat_level: number;
      total_conjunctions: number;
      mitigation_success_rate: number;
      average_response_time: number;
    };
  };
}

const AnalyticsDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState('24h');
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartZoom, setChartZoom] = useState(1);

  const loadData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/analytics/data');
      if (!response.ok) {
        throw new Error('Failed to fetch analytics data');
      }
      const data = await response.json();
      setAnalyticsData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load analytics data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [timeRange]);

  const handleTimeRangeChange = (event: SelectChangeEvent) => {
    setTimeRange(event.target.value);
  };

  const CustomTooltip: React.FC<TooltipProps<ValueType, NameType>> = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1 }}>
          <Typography variant="body2">
            {new Date(label).toLocaleString()}
          </Typography>
          {payload.map((entry) => (
            <Typography key={entry.name} variant="body2" sx={{ color: entry.color }}>
              {String(entry.name)}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
              {String(entry.name).toLowerCase().includes('time') ? 's' : '%'}
            </Typography>
          ))}
        </Paper>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Box>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} md={3} key={i}>
              <Skeleton variant="rectangular" height={120} />
            </Grid>
          ))}
          <Grid item xs={12}>
            <Skeleton variant="rectangular" height={400} />
          </Grid>
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (!analyticsData) {
    return null;
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Header with Time Range Selector */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h4" gutterBottom>
              Protection Analytics
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <MuiTooltip title="Refresh Data">
                <IconButton onClick={loadData} color="primary">
                  <Refresh />
                </IconButton>
              </MuiTooltip>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  onChange={handleTimeRangeChange}
                  label="Time Range"
                  size="small"
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                </Select>
              </FormControl>
            </Box>
          </Box>
        </Grid>

        {/* Summary Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Conjunctions
              </Typography>
              <Typography variant="h4" color="primary">
                {analyticsData.summary.total_conjunctions_analyzed}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Analyzed and Tracked
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Threats Detected
              </Typography>
              <Typography variant="h4" color="warning.main">
                {analyticsData.summary.threats_detected}
              </Typography>
              <Typography variant="body2" color="success.main">
                {analyticsData.summary.threats_mitigated} Mitigated
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Response Time
              </Typography>
              <Typography variant="h4" color="primary">
                {analyticsData.summary.average_response_time.toFixed(2)}s
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Average Response Time
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Protection Coverage
              </Typography>
              <Typography variant="h4" color="success.main">
                {analyticsData.summary.protection_coverage.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Overall Coverage
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Metrics */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Current Protection Metrics
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(analyticsData.current_metrics).map(([key, metric]) => (
                <Grid item xs={12} md={3} key={key}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>
                        {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                      </Typography>
                      <Typography variant="h4" color={metric.status === 'critical' ? 'error' : 'primary'}>
                        {metric.value.toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Trend: {metric.trend}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Trends Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6" gutterBottom>
                Protection Trends
              </Typography>
              <Box>
                <MuiTooltip title="Zoom In">
                  <IconButton
                    onClick={() => setChartZoom(Math.min(2, chartZoom + 0.1))}
                    size="small"
                  >
                    <ZoomIn />
                  </IconButton>
                </MuiTooltip>
                <MuiTooltip title="Zoom Out">
                  <IconButton
                    onClick={() => setChartZoom(Math.max(0.5, chartZoom - 0.1))}
                    size="small"
                  >
                    <ZoomOut />
                  </IconButton>
                </MuiTooltip>
              </Box>
            </Box>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer>
                <LineChart
                  data={analyticsData.trends.daily}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                    scale="time"
                    type="number"
                    domain={['auto', 'auto']}
                  />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="threat_level"
                    name="Threat Level"
                    stroke="#ff4444"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="protection_coverage"
                    name="Protection Coverage"
                    stroke="#00C851"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="response_time"
                    name="Response Time"
                    stroke="#33b5e5"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="mitigation_success"
                    name="Mitigation Success"
                    stroke="#ffbb33"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalyticsDashboard;
