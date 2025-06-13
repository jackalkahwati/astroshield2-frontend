'use client'

import React, { useEffect, useState } from 'react'
import { Paper, Grid, Typography } from '@mui/material'
import {
  ScatterChart,
  LineChart,
  BarChart,
} from '@mui/x-charts'

interface DiagnosticsPayload {
  dual_cdf: { tip: { x: number; y: number }[]; model: { x: number; y: number }[] }
  box_per_inclination: { inclination: string; values: number[] }[]
  dt_scatter: { x: number; y: number }[]
  reliability: { p: number; observed: number }[]
}

const fallback: DiagnosticsPayload = {
  dual_cdf: {
    tip: Array.from({ length: 50 }, (_, i) => ({ x: i / 10, y: i / 50 })),
    model: Array.from({ length: 50 }, (_, i) => ({ x: i / 10, y: (i / 50) ** 1.1 })),
  },
  box_per_inclination: [
    { inclination: '0-10°', values: [10, 20, 15, 25, 30] },
  ],
  dt_scatter: Array.from({ length: 100 }, () => ({ x: Math.random() * 30, y: (Math.random() * 30) - 15 })),
  reliability: Array.from({ length: 10 }, (_, i) => ({ p: (i + 0.5) / 10, observed: (i + 0.5) / 10 })),
}

// Utility to remove any malformed points that could cause chart runtime errors
const sanitizeDiagnostics = (payload: DiagnosticsPayload): DiagnosticsPayload => {
  // Filter helpers
  const isNumber = (n: any): n is number => typeof n === 'number' && !Number.isNaN(n)

  return {
    dual_cdf: {
      tip: payload.dual_cdf.tip.filter((p) => isNumber(p.x) && isNumber(p.y)),
      model: payload.dual_cdf.model.filter((p) => isNumber(p.x) && isNumber(p.y)),
    },
    box_per_inclination: payload.box_per_inclination.filter(
      (b) => typeof b.inclination === 'string' && Array.isArray(b.values) && b.values.length > 0,
    ),
    dt_scatter: payload.dt_scatter.filter((p) => isNumber(p.x) && isNumber(p.y)),
    reliability: payload.reliability.filter((p) => isNumber(p.p) && isNumber(p.observed)),
  }
}

const ChartCard: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <Paper elevation={3} style={{ padding: 16 }}>
    <Typography variant="h6" gutterBottom>
      {title}
    </Typography>
    {children}
  </Paper>
)

export default function DiagnosticsDashboard() {
  const [data, setData] = useState<DiagnosticsPayload | null>(null)

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'
    fetch(`${apiUrl}/api/v1/diagnostics`)
      .then((res) => res.json())
      .then((payload: DiagnosticsPayload) => setData(sanitizeDiagnostics(payload)))
      .catch(() => setData(fallback))
  }, [])

  const diagnostics = data || fallback

  // Log the scatter data just before rendering
  // useEffect(() => {
  //   if (diagnostics && diagnostics.dt_scatter) {
  //     console.log('Scatter data:', JSON.stringify(diagnostics.dt_scatter.slice(0, 5), null, 2)); // Log first 5 points
  //   }
  // }, [diagnostics]);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <ChartCard title="Dual CDF ΔR vs ΔT (Model vs TIP)">
          <LineChart
            height={300}
            series={[
              { data: diagnostics.dual_cdf.tip.map(p => p.y), label: 'TIP', color: '#8884d8' },
              { data: diagnostics.dual_cdf.model.map(p => p.y), label: 'Model', color: '#82ca9d' },
            ]}
            xAxis={[{ data: diagnostics.dual_cdf.tip.map(p => p.x), label: 'Error (km or s)', scaleType: 'linear' }]}
            grid={{ vertical: true, horizontal: true }}
          />
        </ChartCard>
      </Grid>

      <Grid item xs={12} md={6}>
        <ChartCard title="ΔR by Inclination (max whisker shown)">
          <BarChart
            height={300}
            series={[{
              data: diagnostics.box_per_inclination.map(b => Math.max(...b.values)),
              label: 'Max ΔR',
              color: '#1976d2',
            }]}
            xAxis={[{ data: diagnostics.box_per_inclination.map(b => b.inclination), scaleType: 'band' }]}
            // grid={{ vertical: true, horizontal: true }} // Comment out grid
          />
        </ChartCard>
      </Grid>

      <Grid item xs={12} md={6}>
        <ChartCard title="ΔT Scatter vs Days-Before-Decay">
          <ScatterChart
            height={300}
            series={[{ data: diagnostics.dt_scatter, xKey: 'x', yKey: 'y' }]}
            xAxis={[{ label: 'Days before decay', scaleType: 'linear', valueFormatter: (v) => v.toString() }]}
            yAxis={[{ label: 'ΔT (s)' }]}
          />
        </ChartCard>
      </Grid>

      <Grid item xs={12} md={6}>
        <ChartCard title="Reliability Diagram (Predicted vs Observed)">
          <LineChart
            height={300}
            series={[{ data: diagnostics.reliability.map(p => p.observed), label: 'Observed', color: '#ff7300' }]}
            xAxis={[{ data: diagnostics.reliability.map(p => p.p), label: 'Predicted percentile', scaleType: 'linear' }]}
            grid={{ vertical: true, horizontal: true }}
          />
        </ChartCard>
      </Grid>
    </Grid>
  )
} 