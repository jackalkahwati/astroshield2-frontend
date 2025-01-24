"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const generateData = () => {
  const data = []
  for (let i = 0; i < 50; i++) {
    data.push({
      x: Math.random() * 100,
      y: Math.random() * 100,
      z: Math.random() * 100,
    })
  }
  return data
}

const data = generateData()

export function InteractiveVisualization() {
  const [xAxis, setXAxis] = useState("x")
  const [yAxis, setYAxis] = useState("y")

  return (
    <Card>
      <CardHeader>
        <CardTitle>Interactive Visualization: Satellite Positions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between mb-4">
          <Select value={xAxis} onValueChange={setXAxis}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select X-axis" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="x">X Coordinate</SelectItem>
              <SelectItem value="y">Y Coordinate</SelectItem>
              <SelectItem value="z">Z Coordinate</SelectItem>
            </SelectContent>
          </Select>
          <Select value={yAxis} onValueChange={setYAxis}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select Y-axis" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="x">X Coordinate</SelectItem>
              <SelectItem value="y">Y Coordinate</SelectItem>
              <SelectItem value="z">Z Coordinate</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid />
            <XAxis type="number" dataKey={xAxis} name={xAxis.toUpperCase()} />
            <YAxis type="number" dataKey={yAxis} name={yAxis.toUpperCase()} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter name="Satellites" data={data} fill="#8884d8" />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

