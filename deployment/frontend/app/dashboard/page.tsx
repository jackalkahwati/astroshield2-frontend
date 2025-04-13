"use client"

import { useEffect, useState } from "react"
import { ActivityChart } from "@/components/charts/activity-chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { SatelliteIcon, Activity, AlertTriangle, Shield } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { formatDate } from "@/lib/utils/date"

export default function DashboardPage() {
  const [isLoaded, setIsLoaded] = useState(false)
  
  // Sample data
  const satellites = [
    { id: "SAT-001", status: "Active", lastSeen: new Date() },
    { id: "SAT-002", status: "Maintenance", lastSeen: new Date() },
  ]
  
  const activities = [
    { id: 1, message: "Collision Avoided", type: "alert", timestamp: new Date() },
    { id: 2, message: "New Satellite Launched", type: "info", timestamp: new Date() },
  ]

  useEffect(() => {
    // Set loaded after a short delay to ensure all resources are available
    const timer = setTimeout(() => {
      setIsLoaded(true)
    }, 500)
    
    return () => clearTimeout(timer)
  }, [])

  if (!isLoaded) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="text-center">
          <p className="text-xl font-semibold">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <Badge variant="outline" className="text-sm">
          System Status: Operational
        </Badge>
      </div>

      {/* System Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Satellites</CardTitle>
            <SatelliteIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">
              +2 from last month
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">
              -2 from last week
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">98%</div>
            <p className="text-xs text-muted-foreground">
              +2% from last check
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>System Activity</CardTitle>
          <CardDescription>
            Alert frequency over the past week
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[300px]">
          <ActivityChart />
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Recent Satellites */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SatelliteIcon className="h-5 w-5" />
              Recent Satellites
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {satellites.map((sat) => (
                <div
                  key={sat.id}
                  className="flex items-center gap-2 rounded-lg border p-2"
                >
                  <SatelliteIcon className="h-5 w-5 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{sat.id}</p>
                    <p className="text-sm text-muted-foreground">
                      Last seen: {formatDate(sat.lastSeen)}
                    </p>
                  </div>
                  <Badge variant={sat.status === "Active" ? "default" : "secondary"}>
                    {sat.status}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {activities.map((event) => (
                <div
                  key={event.id}
                  className="flex items-center gap-2 rounded-lg border p-2"
                >
                  {event.type === "alert" ? (
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                  ) : (
                    <Activity className="h-5 w-5 text-primary" />
                  )}
                  <div className="flex-1">
                    <p className="font-medium">{event.message}</p>
                    <p className="text-sm text-muted-foreground">
                      {formatDate(event.timestamp)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 