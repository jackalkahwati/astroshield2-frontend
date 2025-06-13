"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"

export function PlanManeuverForm() {
  const [open, setOpen] = useState(false)
  const [satelliteId, setSatelliteId] = useState("")
  const [maneuverType, setManeuverType] = useState("collision_avoidance")
  const [deltaV, setDeltaV] = useState(0.01)
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log("Form submitted:", { satelliteId, maneuverType, deltaV })
    alert("Maneuver planned successfully!")
    setOpen(false)
  }
  
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="bg-blue-600 hover:bg-blue-700">Plan New Maneuver</Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Plan New Maneuver</DialogTitle>
          <DialogDescription>
            Schedule a new orbital maneuver for your satellite.
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="satellite" className="text-sm font-medium">Satellite</label>
            <select 
              id="satellite"
              value={satelliteId}
              onChange={(e) => setSatelliteId(e.target.value)}
              className="w-full p-2 border rounded-md"
              required
            >
              <option value="">Select satellite</option>
              <option value="sat-001">ASTROSHIELD-1</option>
              <option value="sat-002">ASTROSHIELD-2</option>
              <option value="sat-003">SENTINEL-1</option>
              <option value="sat-004">GUARDIAN-1</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label htmlFor="type" className="text-sm font-medium">Maneuver Type</label>
            <select 
              id="type"
              value={maneuverType}
              onChange={(e) => setManeuverType(e.target.value)}
              className="w-full p-2 border rounded-md"
              required
            >
              <option value="collision_avoidance">Collision Avoidance</option>
              <option value="station_keeping">Station Keeping</option>
              <option value="hohmann_transfer">Hohmann Transfer</option>
              <option value="phasing">Phasing Maneuver</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label htmlFor="deltaV" className="text-sm font-medium">Delta-V (m/s)</label>
            <Input
              id="deltaV"
              type="number"
              step="0.001"
              value={deltaV}
              onChange={(e) => setDeltaV(parseFloat(e.target.value) || 0)}
              required
            />
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button type="submit">
              Plan Maneuver
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
} 