"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface SpaceObject {
  id: string
  name: string
  x: number
  y: number
  z: number
}

function calculateDistance(obj1: SpaceObject, obj2: SpaceObject): number {
  const dx = obj1.x - obj2.x
  const dy = obj1.y - obj2.y
  const dz = obj1.z - obj2.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

const initialObjects: SpaceObject[] = [
  { id: "SAT-001", name: "Satellite A", x: 0, y: 0, z: 0 },
  { id: "SAT-002", name: "Satellite B", x: 100, y: 100, z: 100 },
  { id: "DEB-001", name: "Debris 1", x: 50, y: 50, z: 50 },
]

export function ProximityMonitor() {
  const [objects, setObjects] = useState<SpaceObject[]>(initialObjects)
  const [proximityEvents, setProximityEvents] = useState<{ obj1: string; obj2: string; distance: number }[]>([])

  useEffect(() => {
    const interval = setInterval(() => {
      // Update object positions (in a real scenario, this would come from actual tracking data)
      setObjects((prevObjects) =>
        prevObjects.map((obj) => ({
          ...obj,
          x: obj.x + (Math.random() - 0.5) * 10,
          y: obj.y + (Math.random() - 0.5) * 10,
          z: obj.z + (Math.random() - 0.5) * 10,
        })),
      )
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    // Check for proximity events
    const events = []
    for (let i = 0; i < objects.length; i++) {
      for (let j = i + 1; j < objects.length; j++) {
        const distance = calculateDistance(objects[i], objects[j])
        if (distance < 100) {
          // Arbitrary threshold for proximity event
          events.push({ obj1: objects[i].name, obj2: objects[j].name, distance })
        }
      }
    }
    setProximityEvents(events)
  }, [objects])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Proximity Event Monitor</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Object 1</TableHead>
              <TableHead>Object 2</TableHead>
              <TableHead>Distance (km)</TableHead>
              <TableHead>Risk Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {proximityEvents.map((event, index) => (
              <TableRow key={index}>
                <TableCell>{event.obj1}</TableCell>
                <TableCell>{event.obj2}</TableCell>
                <TableCell>{event.distance.toFixed(2)}</TableCell>
                <TableCell>
                  <Badge variant={event.distance < 50 ? "destructive" : event.distance < 75 ? "warning" : "default"}>
                    {event.distance < 50 ? "High" : event.distance < 75 ? "Medium" : "Low"}
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

