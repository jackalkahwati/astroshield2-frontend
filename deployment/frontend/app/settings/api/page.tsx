"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

const endpoints = [
  {
    method: "GET",
    path: "/api/health",
    description: "Health check endpoint",
    response: {
      status: "string",
      timestamp: "ISO string",
      services: {
        database: "string",
        api: "string",
        telemetry: "string"
      }
    }
  },
  {
    method: "GET",
    path: "/api/satellites",
    description: "Get list of all satellites",
    response: "Array of satellite objects"
  },
  {
    method: "GET",
    path: "/api/satellites/{id}",
    description: "Get specific satellite details",
    response: "Detailed satellite object"
  },
  {
    method: "GET",
    path: "/api/telemetry/{id}",
    description: "Get telemetry data for a satellite",
    response: "Telemetry data object"
  },
  {
    method: "GET",
    path: "/api/indicators",
    description: "Get system indicators",
    response: "System indicators object"
  },
  {
    method: "GET",
    path: "/api/stability/{id}",
    description: "Get stability analysis for a satellite",
    response: "Stability analysis object"
  },
  {
    method: "GET",
    path: "/api/comprehensive/data",
    description: "Get comprehensive system data",
    response: "Comprehensive data object"
  },
  {
    method: "GET",
    path: "/api/analytics/data",
    description: "Get analytics data",
    response: "Analytics data object"
  },
  {
    method: "GET",
    path: "/api/maneuvers",
    description: "Get list of maneuvers",
    response: "Array of maneuver objects"
  },
  {
    method: "POST",
    path: "/api/maneuvers",
    description: "Create a new maneuver",
    response: "Created maneuver object"
  }
]

export default function APIPage() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">API Documentation</h2>
          <p className="text-muted-foreground">
            Available endpoints and their responses
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Base URL</CardTitle>
          <CardDescription>
            All API endpoints are prefixed with: <code className="bg-muted px-1 py-0.5 rounded">https://astroshield2-api-production.up.railway.app</code>
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Authentication</CardTitle>
          <CardDescription>
            Currently, the API is open for testing. Authentication will be added in future versions.
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Endpoints</CardTitle>
          <CardDescription>
            Available API endpoints and their details
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[100px]">Method</TableHead>
                <TableHead className="w-[250px]">Path</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="w-[200px]">Response</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {endpoints.map((endpoint) => (
                <TableRow key={`${endpoint.method}-${endpoint.path}`}>
                  <TableCell>
                    <Badge 
                      variant={endpoint.method === "GET" ? "secondary" : "default"}
                    >
                      {endpoint.method}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <code className="bg-muted px-1 py-0.5 rounded">{endpoint.path}</code>
                  </TableCell>
                  <TableCell>{endpoint.description}</TableCell>
                  <TableCell>
                    <code className="bg-muted px-1 py-0.5 rounded text-xs">
                      {typeof endpoint.response === "string" 
                        ? endpoint.response 
                        : "Object"}
                    </code>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Response Format</CardTitle>
          <CardDescription>
            All responses are in JSON format
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium mb-2">Success Response</h4>
              <pre className="bg-muted p-4 rounded-lg text-sm">
                {JSON.stringify({
                  "status": "success",
                  "data": {
                    // Response data here
                  }
                }, null, 2)}
              </pre>
            </div>
            <div>
              <h4 className="text-sm font-medium mb-2">Error Response</h4>
              <pre className="bg-muted p-4 rounded-lg text-sm">
                {JSON.stringify({
                  "status": "error",
                  "error": {
                    "code": "ERROR_CODE",
                    "message": "Error message here"
                  }
                }, null, 2)}
              </pre>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 