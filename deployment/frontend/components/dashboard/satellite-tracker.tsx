"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

const satellites = [
  { id: "SAT-001", name: "AstroShield-1", status: "active", latitude: 40.7128, longitude: -74.006 },
  { id: "SAT-002", name: "AstroShield-2", status: "inactive", latitude: 34.0522, longitude: -118.2437 },
  { id: "SAT-003", name: "AstroShield-3", status: "active", latitude: 51.5074, longitude: -0.1278 },
  { id: "SAT-004", name: "AstroShield-4", status: "active", latitude: 48.8566, longitude: 2.3522 },
]

export function SatelliteTracker() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Satellite ID</TableHead>
          <TableHead>Name</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Latitude</TableHead>
          <TableHead>Longitude</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {satellites.map((satellite) => (
          <TableRow key={satellite.id}>
            <TableCell className="font-medium">{satellite.id}</TableCell>
            <TableCell>{satellite.name}</TableCell>
            <TableCell>
              <Badge variant={satellite.status === "active" ? "default" : "secondary"}>{satellite.status}</Badge>
            </TableCell>
            <TableCell>{satellite.latitude.toFixed(4)}</TableCell>
            <TableCell>{satellite.longitude.toFixed(4)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

