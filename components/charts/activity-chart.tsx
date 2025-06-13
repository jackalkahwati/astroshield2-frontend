"use client"

import { useState, useEffect, useMemo } from "react"
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts"
import { Card, CardContent } from "@/components/ui/card"
import { AlertTriangle, Activity, Info } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { formatChartTime } from "@/lib/utils/date"

// Weekly alert data
const data = [
  { day: "Monday", alerts: 4 },
  { day: "Tuesday", alerts: 3 },
  { day: "Wednesday", alerts: 5 },
  { day: "Thursday", alerts: 2 },
  { day: "Friday", alerts: 6 },
  { day: "Saturday", alerts: 1 },
  { day: "Sunday", alerts: 2 },
]

interface ActivityChartProps {
  type?: "bar" | "line" | "area"
}

export function ActivityChart({ type = "bar" }: ActivityChartProps) {
  const chartComponent = useMemo(() => {
    switch (type) {
      case "line":
        return (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="alerts" stroke="#8884d8" strokeWidth={2} />
          </LineChart>
        )
      case "area":
        return (
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="alerts" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
          </AreaChart>
        )
      default:
        return (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="alerts" fill="#8884d8" />
          </BarChart>
        )
    }
  }, [type])

  return (
    <ResponsiveContainer width="100%" height="100%">
      {chartComponent}
    </ResponsiveContainer>
  )
} 