import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

const satellites = [
  {
    id: "SAT-001",
    name: "StarLink-1234",
    orbitType: "LEO",
    altitude: 549.11,
    velocity: 7.83,
    lastUpdate: "1/23/2025, 7:16:55 PM",
    status: "normal",
  },
  {
    id: "SAT-002",
    name: "GPS-IIIA-06",
    orbitType: "MEO",
    altitude: 20200.18,
    velocity: 3.96,
    lastUpdate: "1/23/2025, 7:16:55 PM",
    status: "warning",
  },
]

export function SatelliteTrackingTable() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="text-white">Object ID</TableHead>
          <TableHead className="text-white">Name</TableHead>
          <TableHead className="text-white">Orbit Type</TableHead>
          <TableHead className="text-white">Altitude (km)</TableHead>
          <TableHead className="text-white">Velocity (km/s)</TableHead>
          <TableHead className="text-white">Last Update</TableHead>
          <TableHead className="text-white">Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {satellites.map((satellite) => (
          <TableRow key={satellite.id}>
            <TableCell className="font-medium text-white">{satellite.id}</TableCell>
            <TableCell className="text-white">{satellite.name}</TableCell>
            <TableCell className="text-white">{satellite.orbitType}</TableCell>
            <TableCell className="text-white">{satellite.altitude.toFixed(2)}</TableCell>
            <TableCell className="text-white">{satellite.velocity.toFixed(2)}</TableCell>
            <TableCell className="text-white">{satellite.lastUpdate}</TableCell>
            <TableCell>
              <Badge
                variant={satellite.status === "normal" ? "default" : "destructive"}
                className={satellite.status === "normal" ? "bg-green-500" : "bg-yellow-500"}
              >
                {satellite.status}
              </Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

