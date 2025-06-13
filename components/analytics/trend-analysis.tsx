"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const generateData = (timeRange: string) => {
  const data = []
  const points = timeRange === "24h" ? 24 : timeRange === "7d" ? 7 : timeRange === "30d" ? 30 : 12
  for (let i = 0; i < points; i++) {
    data.push({
      time: timeRange === "24h" ? `${i}:00` : `Day ${i + 1}`,
      anomalies: Math.floor(Math.random() * 10),
      threats: Math.floor(Math.random() * 5),
      incidents: Math.floor(Math.random() * 3),
    })
  }
  return data
}

interface TrendAnalysisProps {
  timeRange: string
}

export function TrendAnalysis({ timeRange }: TrendAnalysisProps) {
  const [selectedMetric, setSelectedMetric] = useState("anomalies")
  const data = generateData(timeRange)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          Trend Analysis
          <Select value={selectedMetric} onValueChange={setSelectedMetric}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select metric" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="anomalies">Anomalies</SelectItem>
              <SelectItem value="threats">Threats</SelectItem>
              <SelectItem value="incidents">Incidents</SelectItem>
            </SelectContent>
          </Select>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey={selectedMetric} stroke="#8884d8" activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

