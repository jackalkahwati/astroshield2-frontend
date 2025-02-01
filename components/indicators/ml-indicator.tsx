"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronUp } from "lucide-react"

interface MLIndicatorProps {
  title: string
  version: string
  description: string
  confidenceScore: string
  trainingAccuracy: string
  algorithm: string
  passCriteria: string
  features: string[]
}

export function MLIndicator({
  title,
  version,
  description,
  confidenceScore,
  trainingAccuracy,
  algorithm,
  passCriteria,
  features,
}: MLIndicatorProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold text-white">{title}</CardTitle>
          <Badge variant="secondary" className="bg-blue-500 text-white">
            {version}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-300 mb-4">{description}</p>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-sm font-medium text-gray-400">Confidence Score</p>
            <p className="text-2xl font-bold text-green-400">{confidenceScore}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-400">Training Accuracy</p>
            <p className="text-2xl font-bold text-blue-400">{trainingAccuracy}</p>
          </div>
        </div>
        <Button
          variant="outline"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full justify-between text-white border-gray-600"
        >
          {isExpanded ? "Hide Details" : "Show Details"}
          {isExpanded ? <ChevronUp className="ml-2 h-4 w-4" /> : <ChevronDown className="ml-2 h-4 w-4" />}
        </Button>
        {isExpanded && (
          <div className="mt-4 space-y-2">
            <div>
              <p className="text-sm font-medium text-gray-400">Algorithm</p>
              <p className="text-sm text-white">{algorithm}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-400">Pass Criteria</p>
              <p className="text-sm text-white">{passCriteria}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-400">Features</p>
              <ul className="list-disc list-inside text-sm text-white">
                {features.map((feature, index) => (
                  <li key={index}>{feature}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

