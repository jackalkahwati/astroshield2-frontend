"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MLIndicatorList } from "./ml-indicator-list"
import { RuleIndicatorList } from "./rule-indicator-list"
import { ThresholdIndicatorList } from "./threshold-indicator-list"
import { IndicatorSummary } from "./indicator-summary"

export function IndicatorsDashboard() {
  const [activeTab, setActiveTab] = useState("ml")

  return (
    <div className="space-y-4">
      <IndicatorSummary />
      <Tabs defaultValue="ml" className="space-y-4" onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="ml">ML Indicators</TabsTrigger>
          <TabsTrigger value="rule">Rule Indicators</TabsTrigger>
          <TabsTrigger value="threshold">Threshold Indicators</TabsTrigger>
        </TabsList>
        <TabsContent value="ml" className="space-y-4">
          <MLIndicatorList />
        </TabsContent>
        <TabsContent value="rule" className="space-y-4">
          <RuleIndicatorList />
        </TabsContent>
        <TabsContent value="threshold" className="space-y-4">
          <ThresholdIndicatorList />
        </TabsContent>
      </Tabs>
    </div>
  )
}

