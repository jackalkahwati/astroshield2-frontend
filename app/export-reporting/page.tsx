import type { Metadata } from "next"
import { ExportReportingDashboard } from "@/components/export-reporting/dashboard"

export const metadata: Metadata = {
  title: "Data Export & Reporting | AstroShield",
  description: "Export data and generate reports",
}

export default function ExportReportingPage() {
  return (
    <div className="flex-1 space-y-6">
      <h2 className="text-3xl font-bold tracking-tight">Data Export & Reporting</h2>
      <ExportReportingDashboard />
    </div>
  )
}

