import { Metadata } from "next"
import { ComprehensiveDashboard } from "@/components/dashboard/comprehensive"
import { Shell } from "@/components/shell"

export const metadata: Metadata = {
  title: "Dashboard | AstroShield",
  description: "Comprehensive satellite monitoring and control system",
}

export default function DashboardPage() {
  return (
    <Shell>
      <ComprehensiveDashboard />
    </Shell>
  )
}

