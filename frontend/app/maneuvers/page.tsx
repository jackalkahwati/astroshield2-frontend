"use client"

import { useEffect, useState } from "react"
import { format } from "date-fns"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { getManeuvers } from "@/lib/api-client"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { PlanManeuverForm } from "@/components/maneuvers/plan-maneuver-form"
import { Badge } from "@/components/ui/badge"
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

  const formatDate = (dateString: string | undefined) => {
    if (!dateString || isNaN(Date.parse(dateString))) {
      return 'N/A';
    }
    return format(new Date(dateString), 'PPp');
  };

  const renderManeuvers = () => {
    if (!state.data) return null;
    
    return state.data.map((maneuver) => (
      <Card key={maneuver.id} className="mb-4">
        <CardHeader>
          <CardTitle>Maneuver {maneuver.id}</CardTitle>
          <CardDescription>
            Type: {maneuver.type}
            <Badge className="ml-2" variant={maneuver.status === 'completed' ? 'default' : 'secondary'}>
              {maneuver.status}
            </Badge>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2">
            <div>Scheduled: {formatDate(maneuver.scheduledTime)}</div>
            {maneuver.completedTime && <div>Completed: {formatDate(maneuver.completedTime)}</div>}
          </div>
        </CardContent>
      </Card>
    ));
  };

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
        {renderManeuvers()}
      </div>
    </div>
  )
}

