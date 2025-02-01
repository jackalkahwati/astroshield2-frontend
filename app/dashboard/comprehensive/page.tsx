"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { getComprehensiveData } from "@/lib/api-client"
import type { ApiResponse, ApiError, ComprehensiveData } from "@/lib/types"

interface ComprehensiveState {
  data: ComprehensiveData | null
  isLoading: boolean
  error: string | null
}

export default function ComprehensivePage() {
  const [state, setState] = useState<ComprehensiveState>({
    data: null,
    isLoading: true,
    error: null
  })

  useEffect(() => {
    const fetchData = async () => {
      try {
        setState(prev => ({ ...prev, error: null }))
        const response = await getComprehensiveData()
        
        if (!response.data) {
          setState(prev => ({
            ...prev,
            isLoading: false,
            error: response.error?.message || "No data available"
          }))
          return
        }

        setState({
          data: response.data,
          isLoading: false,
          error: null
        })
      } catch (error) {
        console.error("Error fetching comprehensive data:", error)
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: "Failed to load data. Please try again later."
        }))
      }
    }

    fetchData()
  }, [])

  if (state.isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (state.error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {state.error}
        </AlertDescription>
      </Alert>
    )
  }

  if (!state.data) {
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
          <p className="text-2xl font-bold">{state.data.status}</p>
          <p className="text-sm text-muted-foreground">
            Last updated: {new Date(state.data.timestamp).toLocaleString()}
          </p>
        </CardContent>
      </Card>

      {Object.entries(state.data.metrics).map(([key, value]) => (
        <Card key={key}>
          <CardHeader>
            <CardTitle>{key.replace(/_/g, " ").toUpperCase()}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{value.toFixed(1)}%</p>
          </CardContent>
        </Card>
      ))}

      {state.data.alerts.length > 0 && (
        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle>Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {state.data.alerts.map((alert, index) => (
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