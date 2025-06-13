"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Code2, Lock, Unlock } from "lucide-react"

const endpoints = [
  // System Health
  {
    method: "GET",
    path: "/api/health",
    description: "Health check endpoint",
    response: "Object",
    auth: false,
    category: "System"
  },
  
  // TLE Chat & Orbital Intelligence (NEW)
  {
    method: "POST",
    path: "/api/v1/tle/explain",
    description: "Generate natural language explanation for TLE data",
    response: "TLE explanation with analysis",
    auth: true,
    category: "TLE Chat"
  },
  {
    method: "POST",
    path: "/api/v1/tle/intelligence",
    description: "Process orbital intelligence queries with AI models",
    response: "AI-generated orbital intelligence response",
    auth: true,
    category: "TLE Chat"
  },
  {
    method: "GET",
    path: "/api/v1/tle/models",
    description: "Get list of available orbital intelligence models",
    response: "Array of 15 specialized AI models",
    auth: false,
    category: "TLE Chat"
  },
  {
    method: "POST",
    path: "/api/v1/tle/local-inference",
    description: "Run inference using local Hugging Face models",
    response: "Local model analysis results",
    auth: true,
    category: "TLE Chat"
  },

  // ML Model Benchmarking (NEW)
  {
    method: "GET",
    path: "/api/v1/ml/benchmarks",
    description: "Get model benchmark results and performance metrics",
    response: "Benchmark data with scientific validation",
    auth: true,
    category: "ML Models"
  },
  {
    method: "POST",
    path: "/api/v1/ml/tle-local-inference",
    description: "Execute local TLE analysis models offline",
    response: "Local model inference results",
    auth: true,
    category: "ML Models"
  },

  // Traditional Space Operations
  {
    method: "GET",
    path: "/api/satellites",
    description: "Get list of all satellites",
    response: "Array of satellite objects",
    auth: true,
    category: "Space Operations"
  },
  {
    method: "GET",
    path: "/api/satellites/{id}",
    description: "Get specific satellite details",
    response: "Satellite object",
    auth: true,
    category: "Space Operations"
  },
  {
    method: "POST",
    path: "/api/maneuvers",
    description: "Create a new maneuver",
    response: "Created maneuver object",
    auth: true,
    category: "Space Operations"
  },
  {
    method: "GET",
    path: "/api/tracking",
    description: "Get active tracking data",
    response: "Array of tracking objects",
    auth: true,
    category: "Space Operations"
  },
  {
    method: "GET",
    path: "/api/indicators",
    description: "Get all CCDM indicators",
    response: "Array of indicator results",
    auth: true,
    category: "CCDM"
  },
  {
    method: "POST",
    path: "/api/trajectory/analyze",
    description: "Analyze trajectory for reentry",
    response: "Trajectory analysis results",
    auth: true,
    category: "Analysis"
  },
  {
    method: "GET",
    path: "/api/analytics",
    description: "Get analytics dashboard data",
    response: "Analytics summary object",
    auth: true,
    category: "Analytics"
  },
  {
    method: "GET",
    path: "/api/sda/status",
    description: "Get SDA integration status",
    response: "SDA status object",
    auth: true,
    category: "SDA Integration"
  },
  {
    method: "POST",
    path: "/api/sda/udl/collect",
    description: "Submit UDL collection request",
    response: "Collection request confirmation",
    auth: true,
    category: "SDA Integration"
  }
]

const successResponse = {
  "status": "success",
  "data": {
    "satellites": [
      {
        "id": "SAT-001",
        "name": "StarLink-1234",
        "orbit_type": "LEO",
        "altitude_km": 549.11,
        "velocity_km_s": 7.83,
        "status": "active",
        "last_update": "2025-04-22T22:08:00Z"
      }
    ],
    "total": 1,
    "page": 1,
    "limit": 10
  }
}

const tleExplanationResponse = {
  "explanation": "This satellite is in a Low Earth Orbit at approximately 408km altitude with a 51.6° inclination, typical of the International Space Station orbital configuration.",
  "analysis": {
    "orbitalPeriod": "92.8 minutes",
    "inclination": "51.6 degrees",
    "eccentricity": "0.0001",
    "orbitType": "LEO",
    "altitude": 408.5
  },
  "confidence": 0.94,
  "modelUsed": "OrbitAnalyzer-2.0",
  "processingTime": 0.45
}

const orbitalIntelligenceResponse = {
  "response": "Station-keeping maneuver detected with high confidence. Satellite is performing routine orbital maintenance to counteract atmospheric drag.",
  "modelUsed": "ManeuverClassifier-1.5",
  "confidence": 0.91,
  "maneuverType": "station-keeping",
  "threatScore": 2.1,
  "recommendations": [
    "Continue monitoring for completion",
    "Update orbital predictions",
    "Normal operational status"
  ]
}

const errorResponse = {
  "status": "error",
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid authentication token",
    "details": "Please provide a valid Bearer token in the Authorization header"
  }
}

export default function APIDocumentationPage() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">API Documentation</h2>
          <p className="text-muted-foreground mt-1">Complete API reference including TLE Chat & Orbital Intelligence</p>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" className="text-sm">
            <Code2 className="h-3 w-3 mr-1" />
            v1.0
          </Badge>
          <Badge variant="default" className="text-sm bg-cyan-600">
            🤖 AI Powered
          </Badge>
        </div>
      </div>

      {/* Base URL Card */}
      <Card>
        <CardHeader>
          <CardTitle>Base URL</CardTitle>
          <CardDescription>The base URL for all API requests</CardDescription>
        </CardHeader>
        <CardContent>
          <code className="bg-muted px-3 py-2 rounded-md block text-sm">
            https://astroshield2-api-production.up.railway.app
          </code>
        </CardContent>
      </Card>

      {/* New TLE Chat & Orbital Intelligence Section */}
      <Card className="border-cyan-200 bg-cyan-50/50">
        <CardHeader>
          <CardTitle className="text-cyan-900">🚀 NEW: TLE Chat & Orbital Intelligence</CardTitle>
          <CardDescription className="text-cyan-700">
            Advanced AI-powered TLE analysis with 15 specialized orbital intelligence models
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-cyan-900 mb-2">Key Features:</h4>
              <ul className="text-sm text-cyan-800 space-y-1">
                <li>• Natural language TLE explanations</li>
                <li>• 15 specialized AI models (ManeuverClassifier, ThreatScorer, etc.)</li>
                <li>• Local Hugging Face model support</li>
                <li>• Scientific benchmarking (85.6% orbital accuracy)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-cyan-900 mb-2">Available Models:</h4>
              <ul className="text-sm text-cyan-800 space-y-1">
                <li>• OrbitAnalyzer-2.0 (fine-tuned)</li>
                <li>• ManeuverClassifier-1.5</li>
                <li>• ThreatScorer-1.0</li>
                <li>• IntelligenceKernel-1.0</li>
                <li>• + 11 additional specialized models</li>
              </ul>
            </div>
          </div>
          <div className="pt-2">
            <Badge variant="default" className="bg-cyan-600 hover:bg-cyan-700">
              Production Ready
            </Badge>
            <Badge variant="outline" className="ml-2 border-cyan-300 text-cyan-700">
              15 AI Models
            </Badge>
            <Badge variant="outline" className="ml-2 border-cyan-300 text-cyan-700">
              Local ML Support
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Authentication Card */}
      <Card>
        <CardHeader>
          <CardTitle>Authentication</CardTitle>
          <CardDescription>How to authenticate your API requests</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">Current Status</h4>
            <Badge variant="secondary" className="mb-3">
              <Unlock className="h-3 w-3 mr-1" />
              No Authentication Required (Development)
            </Badge>
            <p className="text-sm text-muted-foreground">
              The API is currently open for development purposes. Authentication will be required in production.
            </p>
          </div>
          <div>
            <h4 className="font-medium mb-2">Future Authentication (Coming Soon)</h4>
            <p className="text-sm text-muted-foreground mb-2">
              Bearer token authentication will be required for protected endpoints:
            </p>
            <code className="bg-muted px-3 py-2 rounded-md block text-sm">
              Authorization: Bearer {"<your-api-token>"}
            </code>
          </div>
        </CardContent>
      </Card>

      {/* Endpoints Card */}
      <Card>
        <CardHeader>
          <CardTitle>Endpoints</CardTitle>
          <CardDescription>Available API endpoints and their descriptions</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Category</TableHead>
                <TableHead>Method</TableHead>
                <TableHead>Path</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Auth</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {endpoints.map((endpoint, index) => (
                <TableRow key={index}>
                  <TableCell>
                    <Badge 
                      variant={endpoint.category === "TLE Chat" ? "default" : 
                              endpoint.category === "ML Models" ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {endpoint.category}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge 
                      variant={endpoint.method === "GET" ? "default" : "secondary"}
                      className="font-mono text-xs"
                    >
                      {endpoint.method}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-sm">{endpoint.path}</TableCell>
                  <TableCell className="text-sm">{endpoint.description}</TableCell>
                  <TableCell>
                    {endpoint.auth ? (
                      <Lock className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Unlock className="h-4 w-4 text-muted-foreground" />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Response Format Card */}
      <Card>
        <CardHeader>
          <CardTitle>Response Format</CardTitle>
          <CardDescription>Examples of API response formats</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="satellites" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="satellites">Satellites</TabsTrigger>
              <TabsTrigger value="tle-explanation">TLE Explanation</TabsTrigger>
              <TabsTrigger value="orbital-intelligence">Orbital Intelligence</TabsTrigger>
              <TabsTrigger value="error">Error Response</TabsTrigger>
            </TabsList>
            <TabsContent value="satellites" className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">HTTP Status: 200 OK - Satellites API</h4>
                <p className="text-sm text-muted-foreground mb-3">Response from /api/satellites endpoint</p>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto">
                  <code className="text-sm">
{JSON.stringify(successResponse, null, 2)}
                  </code>
                </pre>
              </div>
            </TabsContent>
            <TabsContent value="tle-explanation" className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">HTTP Status: 200 OK - TLE Explanation</h4>
                <p className="text-sm text-muted-foreground mb-3">Response from /api/v1/tle/explain endpoint</p>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto">
                  <code className="text-sm">
{JSON.stringify(tleExplanationResponse, null, 2)}
                  </code>
                </pre>
              </div>
            </TabsContent>
            <TabsContent value="orbital-intelligence" className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">HTTP Status: 200 OK - Orbital Intelligence</h4>
                <p className="text-sm text-muted-foreground mb-3">Response from /api/v1/tle/intelligence endpoint</p>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto">
                  <code className="text-sm">
{JSON.stringify(orbitalIntelligenceResponse, null, 2)}
                  </code>
                </pre>
              </div>
            </TabsContent>
            <TabsContent value="error" className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">HTTP Status: 401 Unauthorized</h4>
                <p className="text-sm text-muted-foreground mb-3">Standard error response format</p>
                <pre className="bg-muted p-4 rounded-md overflow-x-auto">
                  <code className="text-sm">
{JSON.stringify(errorResponse, null, 2)}
                  </code>
                </pre>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
} 