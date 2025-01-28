"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { RecentSatellites } from "@/components/dashboard/recent-satellites"
import { RecentActivity } from "@/components/dashboard/recent-activity"
import { formatDate } from "@/lib/utils/date"

interface DashboardData {
  metrics: {
    orbit_stability: number
    power_efficiency: number
    thermal_control: number
    communication_quality: number
    protection_coverage: number
  }
  status: string
  alerts: string[]
  timestamp: string
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/dashboard`)
        if (!res.ok) {
          throw new Error(`Failed to fetch dashboard data: ${res.statusText}`)
        }
        const result = await res.json()
        setData(result)
      } catch (err: any) {
        console.error("Error fetching dashboard data:", err)
        setError(err.message)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  if (isLoading) {
    return <div className="p-4">Loading dashboard data...</div>
  }

  if (error) {
    return <div className="p-4 text-destructive">Error: {error}</div>
  }

  return (
    <div className="p-4 space-y-8">
      {/* System Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>System Status</span>
            <Badge variant={data?.status === "operational" ? "default" : "destructive"}>
              {data?.status || "Unknown"}
            </Badge>
          </CardTitle>
          <CardDescription>
            Last updated: {data?.timestamp ? formatDate(data.timestamp) : "N/A"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {data?.metrics && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(data.metrics).map(([key, value]) => (
                <div key={key} className="flex flex-col space-y-1">
                  <span className="text-sm font-medium capitalize">
                    {key.replace(/_/g, " ")}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary rounded-full h-2"
                        style={{ width: `${value}%` }}
                      />
                    </div>
                    <span className="text-sm text-muted-foreground">{value}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alerts Section */}
      {data?.alerts && data.alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {data.alerts.map((alert, index) => (
                <div
                  key={index}
                  className="p-2 bg-muted rounded-lg text-sm flex items-center space-x-2"
                >
                  <span className="w-2 h-2 rounded-full bg-destructive" />
                  <span>{alert}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Activity and Satellites Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <RecentActivity />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Satellites</CardTitle>
          </CardHeader>
          <CardContent>
            <RecentSatellites />
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 