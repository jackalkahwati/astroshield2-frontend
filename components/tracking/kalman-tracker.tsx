"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

// This is a simplified Kalman filter implementation
class KalmanFilter {
  private x: number // state
  private P: number // uncertainty covariance
  private R: number // measurement uncertainty
  private Q: number // process uncertainty

  constructor(initialState: number, initialUncertainty: number) {
    this.x = initialState
    this.P = initialUncertainty
    this.R = 1
    this.Q = 0.1
  }

  predict() {
    this.P = this.P + this.Q
  }

  update(measurement: number) {
    const K = this.P / (this.P + this.R)
    this.x = this.x + K * (measurement - this.x)
    this.P = (1 - K) * this.P
  }

  getState() {
    return this.x
  }
}

export function KalmanTracker() {
  const [filter] = useState(new KalmanFilter(0, 1))
  const [measurements, setMeasurements] = useState<number[]>([])
  const [estimates, setEstimates] = useState<number[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate noisy measurement
      const truePath = Math.sin(Date.now() / 1000)
      const noisyMeasurement = truePath + (Math.random() - 0.5) * 0.5

      filter.predict()
      filter.update(noisyMeasurement)

      setMeasurements((prev) => [...prev, noisyMeasurement].slice(-50))
      setEstimates((prev) => [...prev, filter.getState()].slice(-50))
    }, 100)

    return () => clearInterval(interval)
  }, [filter])

  const data = measurements.map((m, i) => ({
    time: i,
    measurement: m,
    estimate: estimates[i],
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Kalman Filter Tracking</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="measurement" stroke="#8884d8" dot={false} name="Noisy Measurement" />
            <Line type="monotone" dataKey="estimate" stroke="#82ca9d" dot={false} name="Kalman Estimate" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

