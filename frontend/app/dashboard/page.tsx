"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Overview } from "@/components/dashboard/overview"
import { RecentActivity } from "@/components/dashboard/recent-activity"
import { getSystemHealth, getSatellites } from "@/lib/api-client"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { SystemHealth, SatelliteData } from "@/lib/types"

interface DashboardState {
  health: SystemHealth | null
  satellites: SatelliteData[]
  loading: boolean
  error: string | null
}

export default function DashboardPage() {
  const [state, setState] = useState<DashboardState>({
    health: null,
    satellites: [],
    loading: true,
    error: null
  })

  useEffect(() => {
    const fetchData = async () => {
      try {
        setState(prev => ({ ...prev, error: null }))
        const [healthResponse, satellitesResponse] = await Promise.all([
          getSystemHealth(),
          getSatellites()
        ])

        if (healthResponse.error || satellitesResponse.error) {
          setState(prev => ({
            ...prev,
            loading: false,
            error: healthResponse.error?.message || satellitesResponse.error?.message || "Failed to fetch data"
          }))
          return
        }

        setState({
          health: healthResponse.data,
          satellites: satellitesResponse.data || [],
          loading: false,
          error: null
        })
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
        setState(prev => ({
          ...prev,
          loading: false,
          error: "An unexpected error occurred"
        }))
      }
    }

    fetchData()
  }, [])

  if (state.loading) {
    return (
      <div className="flex h-[200px] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-primary"></div>
      </div>
    )
  }

  if (state.error) {
    return (
      <div className="flex h-[200px] items-center justify-center text-destructive">
        {state.error}
      </div>
    )
  }

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
      </div>
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Satellites</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{state.satellites.length}</div>
                <p className="text-xs text-muted-foreground">
                  Active and operational
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">System Health</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{state.health?.status || 'Unknown'}</div>
                <p className="text-xs text-muted-foreground">
                  {state.health?.services?.api || 'Checking status...'}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Response Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">1.2s</div>
                <p className="text-xs text-muted-foreground">
                  Average over last 24h
                </p>
              </CardContent>
            </Card>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Overview</CardTitle>
              </CardHeader>
              <CardContent className="pl-2">
                <Overview />
              </CardContent>
            </Card>
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>
                  System events from the last 24 hours
                </CardDescription>
              </CardHeader>
              <CardContent>
                <RecentActivity />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Activity Log</CardTitle>
              <CardDescription>
                Detailed system activity and events
              </CardDescription>
            </CardHeader>
            <CardContent>
              <RecentActivity />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 