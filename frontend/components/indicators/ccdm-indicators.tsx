"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface Indicator {
  name: string
  value: number
  threshold: number
}

const initialIndicators: Indicator[] = [
  { name: "Orbital Stability", value: 95, threshold: 90 },
  { name: "Collision Risk", value: 2, threshold: 5 },
  { name: "Communication Health", value: 98, threshold: 95 },
  { name: "Power System Efficiency", value: 87, threshold: 80 },
  { name: "Debris Proximity", value: 1, threshold: 3 },
]

export function CCDMIndicators() {
  const [indicators, setIndicators] = useState(initialIndicators)

  useEffect(() => {
    const interval = setInterval(() => {
      setIndicators((prevIndicators) =>
        prevIndicators.map((indicator) => ({
          ...indicator,
          value: Math.min(100, Math.max(0, indicator.value + (Math.random() - 0.5) * 5)),
        })),
      )
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>CCDM Indicators</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {indicators.map((indicator) => (
            <div key={indicator.name} className="space-y-2">
              <div className="flex justify-between">
                <span>{indicator.name}</span>
                <span className={indicator.value >= indicator.threshold ? "text-green-500" : "text-red-500"}>
                  {indicator.value.toFixed(1)}%
                </span>
              </div>
              <Progress
                value={indicator.value}
                className={indicator.value >= indicator.threshold ? "bg-green-200" : "bg-red-200"}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

