"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Shield, AlertTriangle, CheckCircle } from "lucide-react"

export default function ProtectionPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Protection Status</h2>
        <Button>
          <Shield className="mr-2 h-4 w-4" />
          Run Protection Scan
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Protection Level</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">98%</div>
            <p className="text-xs text-muted-foreground">
              Optimal protection active
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Threats</CardTitle>
            <AlertTriangle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2</div>
            <p className="text-xs text-muted-foreground">
              Requiring attention
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Scan</CardTitle>
            <CheckCircle className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2h ago</div>
            <p className="text-xs text-muted-foreground">
              All systems normal
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Protection Details</CardTitle>
          <CardDescription>
            Current protection status and recent activity
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid gap-4">
              <div className="flex items-center justify-between border-b pb-4">
                <div>
                  <p className="font-medium">Debris Avoidance</p>
                  <p className="text-sm text-muted-foreground">Active monitoring and avoidance systems</p>
                </div>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
              <div className="flex items-center justify-between border-b pb-4">
                <div>
                  <p className="font-medium">Collision Prevention</p>
                  <p className="text-sm text-muted-foreground">Real-time trajectory analysis</p>
                </div>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Emergency Response</p>
                  <p className="text-sm text-muted-foreground">Automated response protocols</p>
                </div>
                <CheckCircle className="h-5 w-5 text-success" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 