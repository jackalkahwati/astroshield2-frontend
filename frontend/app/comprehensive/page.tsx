"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { getComprehensiveData } from "@/lib/api-client"
import type { ApiResponse, ApiError, ComprehensiveData } from "@/types"

export default function ComprehensivePage() {
  const [data, setData] = useState<ComprehensiveData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setError(null)
        const response = await getComprehensiveData()
        if (!response.data) {
          setError("No data available")
          return
        }
        // Fix: Extract data from response before setting state
        setData(response.data)
      } catch (error) {
        console.error("Error fetching comprehensive data:", error)
        setError("Failed to load data. Please try again later.")
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {error}
        </AlertDescription>
      </Alert>
    )
  }

  if (!data) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          No data available
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{data.status}</p>
          <p className="text-sm text-muted-foreground">
            Last updated: {new Date(data.timestamp).toLocaleString()}
          </p>
        </CardContent>
      </Card>

      {Object.entries(data.metrics).map(([key, value]) => (
        <Card key={key}>
          <CardHeader>
            <CardTitle>{key.replace(/_/g, " ").toUpperCase()}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{value.toFixed(1)}%</p>
          </CardContent>
        </Card>
      ))}

      {data.alerts.length > 0 && (
        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle>Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {data.alerts.map((alert, index) => (
                <Alert key={index}>
                  <AlertDescription>{alert}</AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 