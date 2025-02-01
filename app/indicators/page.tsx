import type { Metadata } from "next"
import { IndicatorsDashboard } from "@/components/indicators/dashboard"

export const metadata: Metadata = {
  title: "Indicators | AstroShield",
  description: "Key indicators for satellite monitoring and analysis",
}

export default function IndicatorsPage() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight text-white">Indicators</h2>
      </div>
      <IndicatorsDashboard />
    </div>
  )
}

