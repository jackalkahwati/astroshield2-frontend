"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { AlertCircle, AlertTriangle, CheckCircle } from "lucide-react"

const mockGeopoliticalData = {
  USA: { tensionLevel: "Low", allies: ["UK", "France", "Germany"] },
  Russia: { tensionLevel: "High", allies: ["China", "Iran"] },
  China: { tensionLevel: "Medium", allies: ["Russia", "North Korea"] },
  "North Korea": { tensionLevel: "Very High", allies: ["China"] },
}

type Country = keyof typeof mockGeopoliticalData

export function GeopoliticalThreatAnalyzer() {
  const [selectedCountry, setSelectedCountry] = useState<Country | null>(null)
  const [threatAssessment, setThreatAssessment] = useState<string | null>(null)

  const assessThreat = () => {
    if (!selectedCountry) return

    const { tensionLevel, allies } = mockGeopoliticalData[selectedCountry]
    let threatLevel = ""

    switch (tensionLevel) {
      case "Low":
        threatLevel = "Low threat. Continue routine monitoring."
        break
      case "Medium":
        threatLevel = "Moderate threat. Increase vigilance and reporting frequency."
        break
      case "High":
        threatLevel = "High threat. Implement enhanced security measures and prepare contingency plans."
        break
      case "Very High":
        threatLevel = "Critical threat. Activate all defense systems and alert allied nations."
        break
    }

    setThreatAssessment(threatLevel)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Geopolitical Threat Analyzer</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Select onValueChange={(value) => setSelectedCountry(value as Country)}>
            <SelectTrigger>
              <SelectValue placeholder="Select a country" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(mockGeopoliticalData).map((country) => (
                <SelectItem key={country} value={country}>
                  {country}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={assessThreat} disabled={!selectedCountry}>
            Assess Threat
          </Button>
          {threatAssessment && (
            <div className="mt-4 p-4 border rounded-md flex items-start space-x-2">
              {threatAssessment.includes("Low") && <CheckCircle className="text-green-500" />}
              {threatAssessment.includes("Moderate") && <AlertTriangle className="text-yellow-500" />}
              {(threatAssessment.includes("High") || threatAssessment.includes("Critical")) && (
                <AlertCircle className="text-red-500" />
              )}
              <p>{threatAssessment}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

