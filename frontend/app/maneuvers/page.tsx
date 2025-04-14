"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { PlanManeuverForm } from "@/components/maneuvers/plan-maneuver-form"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"

// Define types matching our backend
interface ManeuverDetails {
  delta_v?: number
  duration?: number
  fuel_required?: number
  fuel_used?: number
  target_orbit?: {
    altitude?: number
    inclination?: number
  }
}

interface ManeuverData {
  id: string
  satellite_id: string
  type: string
  status: string
  scheduledTime: string
  completedTime?: string
  created_by?: string
  created_at?: string
  details: ManeuverDetails
}

interface ManeuversState {
  data: ManeuverData[] | null
  isLoading: boolean
  error: string | null
}

// Helper function to format date 
const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
}

function ManeuversFilter({ onFilter }: { onFilter: (query: string) => void }) {
  const [query, setQuery] = useState("")
  
  return (
    <div className="flex space-x-2 items-center mb-4">
      <Input
        className="max-w-sm"
        placeholder="Search maneuvers..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value)
          onFilter(e.target.value)
        }}
      />
    </div>
  )
}

export default function ManeuversPage() {
  const [state, setState] = useState<ManeuversState>({
    data: null,
    isLoading: true,
    error: null,
  })

  const [filteredData, setFilteredData] = useState<ManeuverData[] | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch("/api/v1/maneuvers");
        
        if (!response.ok) {
          throw new Error(`Error fetching maneuvers: ${response.status}`);
        }
        
        const data = await response.json();
        
        setState(prev => ({
          ...prev,
          data: data,
          isLoading: false
        }))
        setFilteredData(data)
      } catch (err) {
        console.error("Failed to fetch maneuvers:", err);
        setState(prev => ({
          ...prev,
          error: err instanceof Error ? err.message : 'Failed to fetch maneuvers data',
          isLoading: false
        }))
      }
    }

    fetchData()
  }, [])

  const handleFilter = (query: string) => {
    if (!state.data) return
    
    if (!query) {
      setFilteredData(state.data)
      return
    }

    const lowercaseQuery = query.toLowerCase()
    const filtered = state.data.filter(maneuver => 
      maneuver.type.toLowerCase().includes(lowercaseQuery) ||
      maneuver.status.toLowerCase().includes(lowercaseQuery) ||
      maneuver.id.toLowerCase().includes(lowercaseQuery)
    )
    setFilteredData(filtered)
  }

  const renderManeuvers = () => {
    if (!filteredData) return null
    
    return filteredData.map((maneuver) => (
      <Card key={maneuver.id} className="mb-4">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{maneuver.type}</span>
            <Badge className={`ml-2 ${getStatusBadgeColor(maneuver.status)}`}>
              {maneuver.status}
            </Badge>
          </CardTitle>
          <CardDescription>ID: {maneuver.id}</CardDescription>
          {maneuver.satellite_id && (
            <CardDescription>Satellite: {maneuver.satellite_id}</CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-2">
          <p>
            <strong>Scheduled:</strong>{" "}
            {maneuver.scheduledTime 
              ? formatDate(maneuver.scheduledTime) 
              : "N/A"}
          </p>
          {maneuver.completedTime && (
            <p>
              <strong>Completed:</strong>{" "}
              {formatDate(maneuver.completedTime)}
            </p>
          )}
          {maneuver.details && (
            <div className="text-sm text-muted-foreground space-y-1">
              <p>Delta-V: {maneuver.details.delta_v?.toFixed(2) ?? "N/A"} m/s</p>
              <p>Duration: {maneuver.details.duration?.toFixed(1) ?? "N/A"} s</p>
              <p>Fuel Required: {maneuver.details.fuel_required?.toFixed(2) ?? "N/A"} kg</p>
              {maneuver.details.fuel_used && (
                <p>Fuel Used: {maneuver.details.fuel_used.toFixed(2)} kg</p>
              )}
              {maneuver.details.target_orbit && (
                <>
                  <p>Target Altitude: {maneuver.details.target_orbit.altitude} km</p>
                  <p>Target Inclination: {maneuver.details.target_orbit.inclination}°</p>
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    ))
  }

  // Helper function to get badge color based on status
  const getStatusBadgeColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-green-500 hover:bg-green-600'
      case 'executing':
        return 'bg-blue-500 hover:bg-blue-600'
      case 'scheduled':
        return 'bg-yellow-500 hover:bg-yellow-600'
      case 'failed':
        return 'bg-red-500 hover:bg-red-600'
      default:
        return '' // default badge color
    }
  }

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
      
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold tracking-tight">Maneuvers</h2>
      </div>

      <ManeuversFilter onFilter={handleFilter} />

      <div className="grid gap-4">
        {filteredData && filteredData.length > 0 ? renderManeuvers() : (
          <div className="text-center p-8 text-muted-foreground">
            {state.data?.length === 0 ? "No maneuvers found." : "No matching maneuvers found."}
          </div>
        )}
      </div>
    </div>
  )
}

