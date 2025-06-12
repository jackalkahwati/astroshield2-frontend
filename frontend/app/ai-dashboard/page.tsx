"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { 
  Brain, Shield, AlertTriangle, Satellite, Activity, 
  Loader2, MessageSquare, Target, BarChart3 
} from "lucide-react"
import { IntentAnalyzer } from "@/components/intent-evaluation/intent-analyzer"

// Hostility Scoring Component
function HostilityScoring() {
  const [actorId, setActorId] = useState("")
  const [satelliteId, setSatelliteId] = useState("")
  const [proximity, setProximity] = useState("")
  const [maneuverCount, setManeuverCount] = useState("")
  const [result, setResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const analyzeHostility = async () => {
    setIsLoading(true)
    setError("")
    setResult(null)

    try {
      const mockHistory = Array(parseInt(maneuverCount) || 1).fill({
        type: "unknown",
        timestamp: new Date().toISOString()
      })

      const response = await fetch("http://localhost:5002/api/v1/ai/hostility-scoring", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actor_id: actorId,
          satellite_id: satelliteId,
          maneuver_history: mockHistory,
          proximity_data: proximity ? {
            minimum_distance: parseFloat(proximity)
          } : undefined
        })
      })

      if (!response.ok) throw new Error(`API error: ${response.status}`)
      
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze")
    } finally {
      setIsLoading(false)
    }
  }

  const getThreatColor = (level: string) => {
    switch (level) {
      case "HIGH": return "bg-red-100 text-red-800"
      case "MEDIUM": return "bg-yellow-100 text-yellow-800"
      case "LOW": return "bg-green-100 text-green-800"
      default: return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          <CardTitle>Hostility Scoring</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>Actor ID</Label>
            <Input
              placeholder="e.g., ADVERSARY-1"
              value={actorId}
              onChange={(e) => setActorId(e.target.value)}
            />
          </div>
          <div>
            <Label>Target Satellite</Label>
            <Input
              placeholder="e.g., SAT-001"
              value={satelliteId}
              onChange={(e) => setSatelliteId(e.target.value)}
            />
          </div>
          <div>
            <Label>Minimum Distance (km)</Label>
            <Input
              type="number"
              placeholder="e.g., 100"
              value={proximity}
              onChange={(e) => setProximity(e.target.value)}
            />
          </div>
          <div>
            <Label>Recent Maneuvers</Label>
            <Input
              type="number"
              placeholder="e.g., 3"
              value={maneuverCount}
              onChange={(e) => setManeuverCount(e.target.value)}
            />
          </div>
        </div>

        <Button
          onClick={analyzeHostility}
          disabled={isLoading || !actorId || !satelliteId}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing Threat...
            </>
          ) : (
            <>
              <Target className="mr-2 h-4 w-4" />
              Analyze Hostility
            </>
          )}
        </Button>

        {error && (
          <div className="p-3 bg-red-50 text-red-600 rounded text-sm">{error}</div>
        )}

        {result && (
          <div className="space-y-4 p-4 border rounded-lg">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold">Threat Assessment</h3>
              <Badge className={getThreatColor(result.threat_level)}>
                {result.threat_level} THREAT
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Hostility Score</p>
                <p className="font-medium">{(result.hostility_score * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-muted-foreground">Confidence</p>
                <p className="font-medium">{(result.confidence * 100).toFixed(0)}%</p>
              </div>
            </div>

            {Object.keys(result.contributing_factors).length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Contributing Factors:</p>
                <div className="space-y-1">
                  {Object.entries(result.contributing_factors).map(([factor, score]) => (
                    <div key={factor} className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        {factor.replace(/_/g, " ")}
                      </span>
                      <span>{((score as number) * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.recommendations.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Recommendations:</p>
                <ul className="space-y-1">
                  {result.recommendations.map((rec: string, idx: number) => (
                    <li key={idx} className="text-sm text-muted-foreground flex items-start">
                      <span className="mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Mini TLE Analyzer Component
function MiniTLEAnalyzer() {
  const [tle1, setTle1] = useState("")
  const [tle2, setTle2] = useState("")
  const [result, setResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const analyzeTLE = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("http://localhost:5002/api/v1/tle-explanations/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          line1: tle1,
          line2: tle2,
          include_risk_assessment: true,
          include_anomaly_detection: true
        })
      })
      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Satellite className="h-5 w-5" />
          <CardTitle>Quick TLE Analysis</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>TLE Line 1</Label>
          <Input
            placeholder="1 25544U 98067A..."
            value={tle1}
            onChange={(e) => setTle1(e.target.value)}
            className="font-mono text-xs"
          />
        </div>
        <div className="space-y-2">
          <Label>TLE Line 2</Label>
          <Input
            placeholder="2 25544  51.6400..."
            value={tle2}
            onChange={(e) => setTle2(e.target.value)}
            className="font-mono text-xs"
          />
        </div>
        <Button
          onClick={analyzeTLE}
          disabled={isLoading || !tle1 || !tle2}
          className="w-full"
        >
          {isLoading ? (
            <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Analyzing...</>
          ) : (
            "Analyze Orbit"
          )}
        </Button>

        {result && (
          <div className="p-3 border rounded-lg space-y-2 text-sm">
            <div>
              <span className="font-medium">{result.orbit_type}</span>
              <span className="text-muted-foreground"> - {result.altitude_description}</span>
            </div>
            <div className="flex justify-between">
              <span>Period: {result.period_minutes.toFixed(1)} min</span>
              <Badge variant={result.decay_risk_level === "HIGH" ? "destructive" : "outline"}>
                {result.decay_risk_level} Risk
              </Badge>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function AIDashboardPage() {
  return (
    <div className="flex-1 space-y-6 p-6">
      <div className="flex items-center gap-3">
        <Brain className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">AI Command Center</h1>
          <p className="text-muted-foreground">Advanced AI-powered space situational awareness</p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">AI Models Active</p>
                <p className="text-2xl font-bold">3</p>
              </div>
              <Activity className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Analyses Today</p>
                <p className="text-2xl font-bold">47</p>
              </div>
              <BarChart3 className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-950 dark:to-amber-900">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Threats Detected</p>
                <p className="text-2xl font-bold">2</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-amber-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="intent" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="intent">Intent Analysis</TabsTrigger>
          <TabsTrigger value="hostility">Hostility Scoring</TabsTrigger>
          <TabsTrigger value="tle">TLE Analysis</TabsTrigger>
          <TabsTrigger value="chat">AI Chat</TabsTrigger>
        </TabsList>

        <TabsContent value="intent" className="space-y-4">
          <IntentAnalyzer />
        </TabsContent>

        <TabsContent value="hostility" className="space-y-4">
          <HostilityScoring />
        </TabsContent>

        <TabsContent value="tle" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <MiniTLEAnalyzer />
            <Card>
              <CardHeader>
                <CardTitle>Recent TLE Analyses</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 border rounded">
                    <span className="text-sm font-medium">ISS (ZARYA)</span>
                    <Badge variant="outline">LEO</Badge>
                  </div>
                  <div className="flex items-center justify-between p-2 border rounded">
                    <span className="text-sm font-medium">STARLINK-1234</span>
                    <Badge variant="outline">LEO</Badge>
                  </div>
                  <div className="flex items-center justify-between p-2 border rounded">
                    <span className="text-sm font-medium">COSMOS 2542</span>
                    <Badge variant="destructive">HIGH RISK</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="chat" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                <CardTitle>AI Space Assistant</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-20" />
                <p>Advanced conversational AI coming soon</p>
                <p className="text-sm mt-2">Chat with AI about orbits, threats, and space operations</p>
                <Button variant="outline" className="mt-4" onClick={() => window.location.href = '/tle-chat'}>
                  Try TLE Analyzer
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 