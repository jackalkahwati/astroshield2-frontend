"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { HighLevelMetrics } from "./high-level-metrics"
import { TrendAnalysis } from "./trend-analysis"
import { PredictiveAnalytics } from "./predictive-analytics"
import { InteractiveVisualization } from "./interactive-visualization"

export function AnalyticsDashboard() {
  const [timeRange, setTimeRange] = useState("24h")

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Analytics Dashboard</CardTitle>
          <select
            className="bg-background border rounded px-3 py-1"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="1y">Last Year</option>
          </select>
        </CardHeader>
        <CardContent>
          <HighLevelMetrics timeRange={timeRange} />
        </CardContent>
      </Card>

      <Tabs defaultValue="trends">
        <TabsList>
          <TabsTrigger value="trends">Trend Analysis</TabsTrigger>
          <TabsTrigger value="predictive">Predictive Analytics</TabsTrigger>
          <TabsTrigger value="interactive">Interactive Visualization</TabsTrigger>
        </TabsList>
        <TabsContent value="trends">
          <TrendAnalysis timeRange={timeRange} />
        </TabsContent>
        <TabsContent value="predictive">
          <PredictiveAnalytics />
        </TabsContent>
        <TabsContent value="interactive">
          <InteractiveVisualization />
        </TabsContent>
      </Tabs>
    </div>
  )
}

