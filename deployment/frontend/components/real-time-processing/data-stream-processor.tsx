"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface DataPoint {
  timestamp: number
  value: number
  source: string
}

const dataSources = ["UDL", "Ground Station", "Satellite Telemetry"]

function generateDataPoint(): DataPoint {
  return {
    timestamp: Date.now(),
    value: Math.random() * 100,
    source: dataSources[Math.floor(Math.random() * dataSources.length)],
  }
}

export function DataStreamProcessor() {
  const [data, setData] = useState<DataPoint[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prevData) => {
        const newData = [...prevData, generateDataPoint()]
        if (newData.length > 50) newData.shift()
        return newData
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Real-Time Data Stream Processor</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()} />
            <YAxis />
            <Tooltip labelFormatter={(label) => new Date(label).toLocaleString()} />
            <Legend />
            {dataSources.map((source, index) => (
              <Line
                key={source}
                type="monotone"
                dataKey="value"
                stroke={`hsl(${index * 120}, 70%, 50%)`}
                name={source}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

