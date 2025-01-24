"use client"

import { DataCard } from "@/components/ui/data-card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { name: "Attitude", value: 95 },
  { name: "Orbit", value: 98 },
  { name: "Thermal", value: 92 },
  { name: "Power", value: 97 },
  { name: "Comms", value: 94 },
]

export function StabilityMetrics() {
  return (
    <DataCard title="Stability Metrics">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis
            stroke="#888888"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </DataCard>
  )
}

