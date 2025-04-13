"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronUp } from "lucide-react"

interface RuleIndicatorProps {
  title: string
  version: string
  description: string
  ruleSet: string
  passCriteria: string
  features: string[]
}

export function RuleIndicator({ title, version, description, ruleSet, passCriteria, features }: RuleIndicatorProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold text-white">{title}</CardTitle>
          <Badge variant="secondary" className="bg-purple-500 text-white">
            {version}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-300 mb-4">{description}</p>
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
              <p className="text-sm font-medium text-gray-400">Rule Set</p>
              <p className="text-sm text-white">{ruleSet}</p>
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

