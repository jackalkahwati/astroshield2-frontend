"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"

const maneuverTypes = ["Hohmann Transfer", "Station Keeping", "Phasing Maneuver", "Collision Avoidance"]

export function ManeuverPlanner() {
  return (
    <Card className="bg-card/50">
      <CardHeader>
        <CardTitle>Plan New Maneuver</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-4">
          <Select>
            <SelectTrigger>
              <SelectValue placeholder="Maneuver Type" />
            </SelectTrigger>
            <SelectContent>
              {maneuverTypes.map((type) => (
                <SelectItem key={type} value={type.toLowerCase()}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="relative">
            <Input type="number" placeholder="0" className="pl-2" />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
              Delta-V (m/s)
            </span>
          </div>

          <Input type="datetime-local" defaultValue="2025-01-23T12:30" />

          <Button className="bg-blue-500 hover:bg-blue-600">PLAN MANEUVER</Button>
        </div>
      </CardContent>
    </Card>
  )
}

