"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Loader2, AlertCircle, CheckCircle, AlertTriangle } from "lucide-react"

interface IntentResult {
  intent_class: string
  confidence: number
  reasoning: string[]
  timestamp: string
}

export function IntentAnalyzer() {
  const [satelliteId, setSatelliteId] = useState("")
  const [maneuverType, setManeuverType] = useState("prograde")
  const [deltaV, setDeltaV] = useState("")
  const [duration, setDuration] = useState("")
  const [result, setResult] = useState<IntentResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleAnalyze = async () => {
    setIsLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("http://localhost:5002/api/v1/ai/intent-classification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          satellite_id: satelliteId,
          maneuver_type: maneuverType,
          delta_v: parseFloat(deltaV),
          duration: parseFloat(duration)
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to analyze intent")
    } finally {
      setIsLoading(false)
    }
  }

  const getIntentIcon = (intent: string) => {
    switch (intent) {
      case "routine_maintenance":
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case "collision_avoidance":
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />
      case "evasion":
        return <AlertCircle className="h-4 w-4 text-red-600" />
      default:
        return null
    }
  }

  const getIntentColor = (intent: string) => {
    switch (intent) {
      case "routine_maintenance":
        return "bg-green-100 text-green-800"
      case "collision_avoidance":
        return "bg-yellow-100 text-yellow-800"
      case "evasion":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Intent Classification</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="satellite-id">Satellite ID</Label>
              <Input
                id="satellite-id"
                placeholder="e.g., SAT-001"
                value={satelliteId}
                onChange={(e) => setSatelliteId(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="maneuver-type">Maneuver Type</Label>
              <Select value={maneuverType} onValueChange={setManeuverType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="prograde">Prograde</SelectItem>
                  <SelectItem value="retrograde">Retrograde</SelectItem>
                  <SelectItem value="radial">Radial</SelectItem>
                  <SelectItem value="normal">Normal</SelectItem>
                  <SelectItem value="stationkeeping">Station-keeping</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="delta-v">Delta-V (m/s)</Label>
              <Input
                id="delta-v"
                type="number"
                placeholder="e.g., 2.5"
                value={deltaV}
                onChange={(e) => setDeltaV(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (seconds)</Label>
              <Input
                id="duration"
                type="number"
                placeholder="e.g., 120"
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
              />
            </div>
          </div>
          
          <Button 
            onClick={handleAnalyze} 
            disabled={isLoading || !satelliteId || !deltaV || !duration}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing Intent...
              </>
            ) : (
              "Analyze Intent"
            )}
          </Button>
          
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}
          
          {result && (
            <div className="mt-4 p-4 border rounded-md space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getIntentIcon(result.intent_class)}
                  <span className="font-semibold capitalize">
                    {result.intent_class.replace(/_/g, " ")}
                  </span>
                </div>
                <Badge className={getIntentColor(result.intent_class)}>
                  {(result.confidence * 100).toFixed(0)}% Confidence
                </Badge>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm font-medium">AI Reasoning:</p>
                <ul className="text-sm text-gray-600 space-y-1">
                  {result.reasoning.map((reason, idx) => (
                    <li key={idx} className="flex items-start">
                      <span className="mr-2">•</span>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
              
              <p className="text-xs text-gray-500">
                Analysis timestamp: {new Date(result.timestamp).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

