"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

const data = [
  { name: "Satellite A", predicted: 65, actual: 70 },
  { name: "Satellite B", predicted: 45, actual: 48 },
  { name: "Satellite C", predicted: 80, actual: 77 },
  { name: "Satellite D", predicted: 55, actual: 59 },
  { name: "Satellite E", predicted: 70, actual: 72 },
]

export function PredictiveAnalytics() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Predictive Analytics: Satellite Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="predicted" fill="#8884d8" name="Predicted Performance" />
            <Bar dataKey="actual" fill="#82ca9d" name="Actual Performance" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

