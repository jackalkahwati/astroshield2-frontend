"use client"

import { useState } from "react"
import type { Metadata } from "next"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Satellite, Send } from "lucide-react"

// Type definitions
interface TLEExplanation {
  norad_id: string
  satellite_name: string
  orbit_description: string
  orbit_type: string
  altitude_description: string
  period_minutes: number
  inclination_degrees: number
  eccentricity: number
  decay_risk_score: number
  decay_risk_level: string
  anomaly_flags: string[]
  predicted_lifetime_days?: number
  confidence_score: number
}

export default function TLEChatPage() {
  const [line1, setLine1] = useState("")
  const [line2, setLine2] = useState("")
  const [satelliteName, setSatelliteName] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [explanation, setExplanation] = useState<TLEExplanation | null>(null)
  const [error, setError] = useState("")

  // Example TLE for ISS
  const loadExample = () => {
    setSatelliteName("ISS (ZARYA)")
    setLine1("1 25544U 98067A   24079.07757601 .00016717 00000+0 10270-3 0  9994")
    setLine2("2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263")
  }

  const analyzeTLE = async () => {
    setIsLoading(true)
    setError("")
    setExplanation(null)

    try {
      const response = await fetch("http://localhost:5002/api/v1/tle-explanations/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          satellite_name: satelliteName || "Unknown Satellite",
          line1: line1.trim(),
          line2: line2.trim(),
          include_risk_assessment: true,
          include_anomaly_detection: true
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setExplanation(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze TLE")
    } finally {
      setIsLoading(false)
    }
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case "LOW": return "text-green-600"
      case "MEDIUM": return "text-yellow-600"
      case "HIGH": return "text-red-600"
      default: return "text-gray-600"
    }
  }

  return (
    <div className="flex-1 space-y-6 p-6">
      <div className="flex items-center gap-2">
        <Satellite className="h-6 w-6" />
        <h2 className="text-3xl font-bold">TLE Orbit Analyzer</h2>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Enter TLE Data</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="satellite-name">Satellite Name (Optional)</Label>
              <Input
                id="satellite-name"
                placeholder="e.g., ISS (ZARYA)"
                value={satelliteName}
                onChange={(e) => setSatelliteName(e.target.value)}
              />
            </div>

            <div>
              <Label htmlFor="line1">TLE Line 1</Label>
              <Textarea
                id="line1"
                placeholder="1 25544U 98067A..."
                value={line1}
                onChange={(e) => setLine1(e.target.value)}
                className="font-mono text-sm"
                rows={2}
              />
            </div>

            <div>
              <Label htmlFor="line2">TLE Line 2</Label>
              <Textarea
                id="line2"
                placeholder="2 25544  51.6400..."
                value={line2}
                onChange={(e) => setLine2(e.target.value)}
                className="font-mono text-sm"
                rows={2}
              />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={analyzeTLE}
                disabled={isLoading || !line1 || !line2}
                className="flex-1"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send className="mr-2 h-4 w-4" />
                    Analyze Orbit
                  </>
                )}
              </Button>
              <Button variant="outline" onClick={loadExample}>
                Load Example
              </Button>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            {explanation ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-lg">
                    {explanation.satellite_name} (NORAD: {explanation.norad_id})
                  </h3>
                  <p className="text-muted-foreground mt-1">
                    {explanation.orbit_type} - {explanation.orbit_description}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Altitude</p>
                    <p className="font-medium">{explanation.altitude_description}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Period</p>
                    <p className="font-medium">{explanation.period_minutes.toFixed(1)} minutes</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Inclination</p>
                    <p className="font-medium">{explanation.inclination_degrees.toFixed(1)}°</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Eccentricity</p>
                    <p className="font-medium">{explanation.eccentricity.toFixed(4)}</p>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-semibold mb-2">Decay Risk Assessment</h4>
                  <div className="flex items-center justify-between">
                    <span className={`font-medium ${getRiskColor(explanation.decay_risk_level)}`}>
                      {explanation.decay_risk_level} RISK
                    </span>
                    <span className="text-sm text-muted-foreground">
                      Score: {(explanation.decay_risk_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  {explanation.predicted_lifetime_days && (
                    <p className="text-sm mt-1">
                      Estimated lifetime: {explanation.predicted_lifetime_days.toFixed(0)} days
                    </p>
                  )}
                </div>

                {explanation.anomaly_flags.length > 0 && (
                  <div className="border-t pt-4">
                    <h4 className="font-semibold mb-2">Anomalies Detected</h4>
                    <div className="space-y-1">
                      {explanation.anomaly_flags.map((flag, idx) => (
                        <div key={idx} className="text-sm text-yellow-600">
                          ⚠️ {flag.replace(/_/g, " ")}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="text-xs text-muted-foreground pt-2">
                  Confidence: {(explanation.confidence_score * 100).toFixed(0)}%
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <Satellite className="h-12 w-12 mx-auto mb-4 opacity-20" />
                <p>Enter TLE data to see orbit analysis</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 