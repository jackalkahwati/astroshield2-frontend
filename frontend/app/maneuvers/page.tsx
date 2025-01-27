"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { getManeuvers } from "@/lib/api-client"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { PlanManeuverForm } from "@/components/maneuvers/plan-maneuver-form"
import { Badge } from "@/components/ui/badge"
import { format } from "date-fns"
import type { ManeuverData } from "@/lib/types"

interface ManeuversState {
  data: ManeuverData[] | null
  isLoading: boolean
  error: string | null
}

export default function ManeuversPage() {
  const [state, setState] = useState<ManeuversState>({
    data: null,
    isLoading: true,
    error: null,
  })

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await getManeuvers()
        if (!response.data) {
          setState(prev => ({
            ...prev,
            error: response.error?.message || 'No maneuvers data available',
            isLoading: false
          }))
          return
        }
        setState(prev => ({
          ...prev,
          data: response.data,
          isLoading: false
        }))
      } catch (err) {
        setState(prev => ({
          ...prev,
          error: 'Failed to fetch maneuvers data',
          isLoading: false
        }))
      }
    }

    fetchData()
  }, [])

  if (state.error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{state.error}</AlertDescription>
      </Alert>
    )
  }

  if (state.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-[200px] w-full" />
        <Skeleton className="h-[200px] w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <PlanManeuverForm />
      
      <div className="grid gap-4">
        {state.data?.map((maneuver) => (
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
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

