"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import { ZoomIn, ZoomOut } from "lucide-react"
import { Button } from "@/components/ui/button"

const data = [
  { time: "00:00", threat: 82, coverage: 91, response: 1.4, mitigation: 88 },
  { time: "04:00", threat: 86, coverage: 92, response: 1.3, mitigation: 90 },
  { time: "08:00", threat: 84, coverage: 92, response: 1.2, mitigation: 92 },
  { time: "12:00", threat: 88, coverage: 93, response: 1.3, mitigation: 91 },
  { time: "16:00", threat: 85, coverage: 92, response: 1.4, mitigation: 89 },
  { time: "20:00", threat: 83, coverage: 92, response: 1.3, mitigation: 90 },
  { time: "24:00", threat: 85, coverage: 93, response: 1.2, mitigation: 92 },
]

export function ProtectionTrends() {
  return (
    <Card className="bg-card/50">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Protection Trends</CardTitle>
        <div className="flex gap-2">
          <Button variant="outline" size="icon">
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon">
            <ZoomOut className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            threat: {
              label: "Threat Level",
              color: "hsl(var(--destructive))",
            },
            coverage: {
              label: "Protection Coverage",
              color: "hsl(var(--success))",
            },
            response: {
              label: "Response Time",
              color: "hsl(var(--blue))",
            },
            mitigation: {
              label: "Mitigation Success",
              color: "hsl(var(--warning))",
            },
          }}
          className="h-[400px]"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <XAxis dataKey="time" stroke="#888888" fontSize={12} />
              <YAxis stroke="#888888" fontSize={12} />
              <ChartTooltip />
              <Line type="monotone" dataKey="threat" stroke="var(--color-threat)" strokeWidth={2} />
              <Line type="monotone" dataKey="coverage" stroke="var(--color-coverage)" strokeWidth={2} />
              <Line type="monotone" dataKey="response" stroke="var(--color-response)" strokeWidth={2} />
              <Line type="monotone" dataKey="mitigation" stroke="var(--color-mitigation)" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

