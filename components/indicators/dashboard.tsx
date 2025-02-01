"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MLIndicatorList } from "./ml-indicator-list"
import { RuleIndicatorList } from "./rule-indicator-list"
import { ThresholdIndicatorList } from "./threshold-indicator-list"
import { IndicatorSummary } from "./indicator-summary"
import axios from "axios"

const API_BASE = "https://nosy-boy-production.up.railway.app/api/v1"

interface UDLData {
  satellites: Array<{
    id: string;
    indicators: Record<string, any>;
  }>;
}

interface StabilityData {
  status: string;
  metrics: Record<string, any>;
}

interface IndicatorsData {
  udl: UDLData;
  stability: StabilityData;
}

export function IndicatorsDashboard() {
  const [activeTab, setActiveTab] = useState("ml")
  const [indicatorsData, setIndicatorsData] = useState<IndicatorsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [udlResponse, stabilityResponse] = await Promise.all([
          axios.get<UDLData>(`${API_BASE}/udl/data`),
          axios.get<StabilityData>(`${API_BASE}/indicators/stability/latest`)
        ])

        setIndicatorsData({
          udl: udlResponse.data,
          stability: stabilityResponse.data
        })
      } catch (error) {
        console.error("Error fetching indicators data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>
  }

  return (
    <div className="space-y-4">
      <IndicatorSummary data={indicatorsData} />
      <Tabs defaultValue="ml" className="space-y-4" onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="ml">ML Indicators</TabsTrigger>
          <TabsTrigger value="rule">Rule Indicators</TabsTrigger>
          <TabsTrigger value="threshold">Threshold Indicators</TabsTrigger>
        </TabsList>
        <TabsContent value="ml" className="space-y-4">
          <MLIndicatorList data={indicatorsData} />
        </TabsContent>
        <TabsContent value="rule" className="space-y-4">
          <RuleIndicatorList data={indicatorsData} />
        </TabsContent>
        <TabsContent value="threshold" className="space-y-4">
          <ThresholdIndicatorList data={indicatorsData} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

