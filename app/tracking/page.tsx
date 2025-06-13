import type { Metadata } from "next"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { SatelliteTrackingDashboard } from "@/components/satellite-tracking/dashboard"

export const metadata: Metadata = {
  title: "Satellite Tracking | AstroShield",
  description: "Real-time satellite tracking and monitoring",
}

export default function SatelliteTrackingPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Satellite Tracking</h1>
      </div>
      <SatelliteTrackingDashboard />
    </div>
  )
}

