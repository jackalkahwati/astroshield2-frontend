import type { Metadata } from "next"
import { AnalyticsDashboard } from "@/components/analytics/dashboard"

export const metadata: Metadata = {
  title: "Analytics | AstroShield",
  description: "Advanced analytics and insights",
}

export default function AnalyticsPage() {
  return (
    <div className="flex-1 space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Advanced Analytics</h2>
      <AnalyticsDashboard />
    </div>
  )
}

