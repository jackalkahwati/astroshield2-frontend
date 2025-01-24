"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AdvancedCharacterization } from "@/components/characterization/advanced-characterization"
import { KalmanTracker } from "@/components/tracking/kalman-tracker"
import { GeopoliticalThreatAnalyzer } from "@/components/threat-assessment/geopolitical-threat-analyzer"
import { IntentAnalyzer } from "@/components/intent-evaluation/intent-analyzer"
import { AdvancedManeuverDetector } from "@/components/maneuver-detection/advanced-maneuver-detector"
import { ProximityMonitor } from "@/components/proximity-monitoring/proximity-monitor"
import { KafkaIntegration } from "@/components/pubsub/kafka-integration"
import { DataStreamProcessor } from "@/components/real-time-processing/data-stream-processor"
import { CCDMIndicators } from "@/components/indicators/ccdm-indicators"

export function CCDMDashboard() {
  const [activeTab, setActiveTab] = useState("overview")

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>CCDM Advanced Dashboard</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3 lg:grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="characterization">Characterization</TabsTrigger>
            <TabsTrigger value="tracking">Tracking</TabsTrigger>
            <TabsTrigger value="threats">Threats</TabsTrigger>
            <TabsTrigger value="realtime">Real-time Data</TabsTrigger>
          </TabsList>
          <TabsContent value="overview">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <CCDMIndicators />
              <ProximityMonitor />
              <KafkaIntegration />
            </div>
          </TabsContent>
          <TabsContent value="characterization">
            <AdvancedCharacterization />
          </TabsContent>
          <TabsContent value="tracking">
            <div className="space-y-4">
              <KalmanTracker />
              <AdvancedManeuverDetector />
            </div>
          </TabsContent>
          <TabsContent value="threats">
            <div className="space-y-4">
              <GeopoliticalThreatAnalyzer />
              <IntentAnalyzer />
            </div>
          </TabsContent>
          <TabsContent value="realtime">
            <DataStreamProcessor />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

