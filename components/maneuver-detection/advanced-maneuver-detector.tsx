"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Badge } from "@/components/ui/badge"

// Mock function to simulate satellite position data
function generateSatelliteData(time: number) {
  const baseOrbit = Math.sin(time / 10) * 100 + 500
  const maneuver = time > 50 && time < 70 ? (time - 50) * 5 : 0
  return baseOrbit + maneuver + (Math.random() - 0.5) * 10
}

export function AdvancedManeuverDetector() {
  const [data, setData] = useState<{ time: number; position: number }[]>([])
  const [maneuverDetected, setManeuverDetected] = useState(false)

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prevData) => {
        const newData = [...prevData, { time: prevData.length, position: generateSatelliteData(prevData.length) }]
        if (newData.length > 100) newData.shift()

        // Simple maneuver detection logic
        if (newData.length > 10) {
          const recentData = newData.slice(-10)
          const avgChange =
            recentData.reduce((sum, point, index, array) => {
              if (index === 0) return sum
              return sum + Math.abs(point.position - array[index - 1].position)
            }, 0) / 9

          setManeuverDetected(avgChange > 10)
        }

        return newData
      })
    }, 200)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          Advanced Maneuver Detection
          {maneuverDetected && <Badge variant="destructive">Maneuver Detected</Badge>}
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
            <Line type="monotone" dataKey="position" stroke="#8884d8" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

