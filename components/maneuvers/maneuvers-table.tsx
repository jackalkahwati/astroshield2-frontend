"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type ManeuverStatus = "EXECUTING" | "FAILED" | "COMPLETED"

interface Maneuver {
  id: string
  type: string
  deltaV: number
  executionTime: string
  duration: number
  status: ManeuverStatus
}

const maneuvers: Maneuver[] = [
  {
    id: "MNV-6460",
    type: "hohmann",
    deltaV: 1.12,
    executionTime: "1/23/2025, 7:47:32 PM",
    duration: 2136,
    status: "EXECUTING",
  },
  {
    id: "MNV-6295",
    type: "collision",
    deltaV: 0.16,
    executionTime: "1/23/2025, 12:47:32 PM",
    duration: 2763,
    status: "FAILED",
  },
  {
    id: "MNV-2130",
    type: "collision",
    deltaV: 0.25,
    executionTime: "1/23/2025, 5:47:32 AM",
    duration: 6893,
    status: "EXECUTING",
  },
  {
    id: "MNV-5233",
    type: "phasing",
    deltaV: 1.45,
    executionTime: "1/23/2025, 1:47:32 AM",
    duration: 7123,
    status: "FAILED",
  },
  {
    id: "MNV-4736",
    type: "collision",
    deltaV: 0.88,
    executionTime: "1/22/2025, 8:47:32 PM",
    duration: 6149,
    status: "COMPLETED",
  },
]

export function ManeuversTable() {
  return (
    <Card className="bg-card/50">
      <CardHeader>
        <CardTitle>Active Maneuvers</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Delta-V (m/s)</TableHead>
              <TableHead>Execution Time</TableHead>
              <TableHead>Duration (s)</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {maneuvers.map((maneuver) => (
              <TableRow key={maneuver.id}>
                <TableCell className="font-medium">{maneuver.id}</TableCell>
                <TableCell>{maneuver.type}</TableCell>
                <TableCell>{maneuver.deltaV}</TableCell>
                <TableCell>{maneuver.executionTime}</TableCell>
                <TableCell>{maneuver.duration}</TableCell>
                <TableCell>
                  <Badge
                    className={cn({
                      "bg-warning text-warning-foreground": maneuver.status === "EXECUTING",
                      "bg-destructive text-destructive-foreground": maneuver.status === "FAILED",
                      "bg-success text-success-foreground": maneuver.status === "COMPLETED",
                    })}
                  >
                    {maneuver.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

