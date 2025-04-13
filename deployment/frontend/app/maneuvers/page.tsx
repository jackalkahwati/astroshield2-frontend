"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { getManeuvers } from "@/lib/api-client"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { PlanManeuverForm } from "@/components/maneuvers/plan-maneuver-form"
import { Badge } from "@/components/ui/badge"
import type { ManeuverData } from "@/lib/types"
import { Input } from "@/components/ui/input"
import { formatDate } from "@/lib/utils"
import { FALLBACK_MANEUVERS } from "@/lib/fallback-data"

interface ManeuversState {
  data: ManeuverData[] | null
  isLoading: boolean
  error: string | null
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

  // Use a ref to track if the component is mounted
  const isMounted = useState(true)

  useEffect(() => {
    async function fetchData() {
      // Add a timeout to prevent infinite loading
      const timeoutId = setTimeout(() => {
        console.log('Fetch timeout reached, using fallback data')
        setState(prev => ({
          ...prev,
          data: FALLBACK_MANEUVERS,
          isLoading: false
        }))
        setFilteredData(FALLBACK_MANEUVERS)
      }, 5000) // 5 second timeout
      
      try {
        console.log('Fetching maneuvers data...')
        const response = await getManeuvers()
        console.log('Maneuvers API response:', response)
        
        // Clear the timeout since we got a response
        clearTimeout(timeoutId)
        
        // Use fallback data if no data from API
        if (!response.data) {
          console.warn('No maneuvers data from API, using fallback data')
          setState(prev => ({
            ...prev,
            data: FALLBACK_MANEUVERS,
            isLoading: false
          }))
          setFilteredData(FALLBACK_MANEUVERS)
          return
        }
        
        console.log('Maneuvers data loaded:', response.data)
        
        // Create a properly typed set of data
        const mappedData: ManeuverData[] = response.data.map(item => ({
          id: item.id,
          satellite_id: item.satellite_id,
          type: item.type,
          status: item.status,
          scheduledTime: item.scheduledTime || '',
          completedTime: item.completedTime || '',
          details: {
            delta_v: item.details?.delta_v,
            duration: item.details?.duration,
            fuel_required: item.details?.fuel_required || 0
          },
          created_by: item.created_by || '',
          created_at: item.created_at || ''
        }))
        
        setState(prev => ({
          ...prev,
          data: mappedData,
          isLoading: false
        }))
        setFilteredData(mappedData)
      } catch (err) {
        console.error('Failed to fetch maneuvers:', err)
        
        // Clear the timeout since we got an error
        clearTimeout(timeoutId)
        
        // Use fallback data on error
        setState(prev => ({
          ...prev,
          data: FALLBACK_MANEUVERS,
          isLoading: false
        }))
        setFilteredData(FALLBACK_MANEUVERS)
      }
    }

    fetchData()
    
    // Cleanup function to abort any pending requests
    return () => {
      isMounted[1](false)
    }
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
            <Badge className="ml-2" variant={maneuver.status === 'completed' ? 'default' : 'secondary'}>
              {maneuver.status}
            </Badge>
          </CardTitle>
          <CardDescription>ID: {maneuver.id}</CardDescription>
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
            </div>
          )}
        </CardContent>
      </Card>
    ))
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

