"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { getManeuvers } from "@/lib/api-client"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { PlanManeuverForm } from "@/components/maneuvers/plan-maneuver-form"
import { Badge } from "@/components/ui/badge"
import { format } from "date-fns"
import { ManeuverData, ApiResponse } from '@/lib/types'

interface ManeuversResponse {
  maneuvers: ManeuverData[]
  resources: {
    fuel_remaining: number
    thrust_capacity: number
    next_maintenance: string
  }
  lastUpdate: string
}

interface ManeuverDetails {
  deltaV: number
  duration: number
  fuel_required: number
  rotation_angle: number
  fuel_used?: number
}

interface Maneuver {
  id: string
  type: string
  status: string
  scheduledTime: string
  completedTime?: string
  details: ManeuverDetails
}

export default function ManeuversPage() {
  const [data, setData] = useState<ManeuversResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await getManeuvers()
        setData(response.data)
      } catch (err) {
        setError('Failed to fetch maneuvers data')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  if (!data) return <div>No data available</div>

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Maneuvers</h2>
          <p className="text-muted-foreground">
            Plan and monitor orbital maneuvers
          </p>
        </div>
        <PlanManeuverForm />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Fuel Status</CardTitle>
            <CardDescription>Current fuel levels and capacity</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.resources.fuel_remaining.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Next maintenance: {format(new Date(data.resources.next_maintenance), "PPP")}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Thrust Capacity</CardTitle>
            <CardDescription>Available thrust for maneuvers</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.resources.thrust_capacity.toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Last Update</CardTitle>
            <CardDescription>System status update time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {format(new Date(data.lastUpdate), "HH:mm:ss")}
            </div>
            <p className="text-xs text-muted-foreground">
              {format(new Date(data.lastUpdate), "PPP")}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4">
        {data.maneuvers.map((maneuver) => (
          <Card key={maneuver.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle>{maneuver.type}</CardTitle>
                  <CardDescription>ID: {maneuver.id}</CardDescription>
                </div>
                <Badge
                  variant={
                    maneuver.status === "completed"
                      ? "default"
                      : maneuver.status === "failed"
                      ? "destructive"
                      : "secondary"
                  }
                >
                  {maneuver.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <div className="text-sm font-medium">Scheduled Time</div>
                  <div>{format(new Date(maneuver.scheduledTime), "PPP HH:mm")}</div>
                </div>
                {maneuver.completedTime && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Completed Time</div>
                    <div>{format(new Date(maneuver.completedTime), "PPP HH:mm")}</div>
                  </div>
                )}
                <div className="space-y-2">
                  <div className="text-sm font-medium">Delta-V</div>
                  <div>{maneuver.details.delta_v?.toFixed(2)} m/s</div>
                </div>
                <div className="space-y-2">
                  <div className="text-sm font-medium">Duration</div>
                  <div>{maneuver.details.duration?.toFixed(2)} s</div>
                </div>
                <div className="space-y-2">
                  <div className="text-sm font-medium">Fuel Required</div>
                  <div>{maneuver.details.fuel_required?.toFixed(2)} kg</div>
                </div>
                {maneuver.details.target_orbit && (
                  <>
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Target Altitude</div>
                      <div>{maneuver.details.target_orbit.altitude?.toFixed(2)} km</div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Target Inclination</div>
                      <div>{maneuver.details.target_orbit.inclination?.toFixed(2)}°</div>
                    </div>
                  </>
                )}
                {maneuver.details.fuel_used && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Fuel Used</div>
                    <div>{maneuver.details.fuel_used.toFixed(2)} kg</div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

