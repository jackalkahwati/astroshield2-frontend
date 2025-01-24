import type { Metadata } from "next"
import { ManeuverPlanner } from "@/components/maneuvers/maneuver-planner"
import { ManeuversTable } from "@/components/maneuvers/maneuvers-table"

export const metadata: Metadata = {
  title: "Maneuvers | AstroShield",
  description: "Plan and monitor satellite maneuvers",
}

export default function ManeuversPage() {
  return (
    <div className="flex-1 space-y-6">
      <ManeuverPlanner />
      <ManeuversTable />
    </div>
  )
}

