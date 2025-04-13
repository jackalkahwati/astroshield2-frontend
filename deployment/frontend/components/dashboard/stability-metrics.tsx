"use client"

import { DashboardCard } from "@/components/dashboard/dashboard-card"
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts"

const data = [
  { name: "Attitude", value: 85 },
  { name: "Orbit", value: 92 },
  { name: "Thermal", value: 78 },
  { name: "Power", value: 88 },
]

export function StabilityMetrics() {
  return (
    <DashboardCard title="Stability Metrics">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
          <Tooltip />
          <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </DashboardCard>
  )
}

