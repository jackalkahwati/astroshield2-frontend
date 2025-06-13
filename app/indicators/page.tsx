"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChevronDown, ChevronUp, Brain, Scale, AlertTriangle } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { 
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger 
} from "@/components/ui/collapsible"

// Define all 19 indicators from the specification
const mlIndicators = [
  {
    name: "Object Stability",
    version: "ML v2.3.1",
    confidence: 97.0,
    accuracy: 96.0,
    description: "Evaluates if an object is maintaining stable orbit and behavior",
    features: ["Orbital Elements", "Attitude Control", "Power Systems"],
    ruleSet: "Stability Threshold Rules",
    passCriteria: "Stability score > 0.85"
  },
  {
    name: "Stability Change",
    version: "ML v2.1.1",
    confidence: 95.0,
    accuracy: 94.0,
    description: "Detects changes in stability patterns over time",
    features: ["Historical Data", "Rate of Change", "Anomaly Detection"],
    ruleSet: "Change Detection Algorithm",
    passCriteria: "Change magnitude < 0.1 over 24h"
  },
  {
    name: "Maneuvers Detected",
    version: "ML v3.0.1",
    confidence: 96.0,
    accuracy: 95.0,
    description: "Identifies and classifies orbital maneuvers",
    features: ["Delta-V", "Thrust Profile", "Orbital Changes"],
    ruleSet: "Maneuver Classification",
    passCriteria: "Authorized maneuvers only"
  },
  {
    name: "Pattern of Life",
    version: "ML v2.8.0",
    confidence: 93.0,
    accuracy: 92.0,
    description: "Analyzes behavioral patterns and detects anomalies",
    features: ["Temporal Patterns", "Activity Cycles", "Deviation Analysis"],
    ruleSet: "Behavioral Baseline",
    passCriteria: "Within expected patterns"
  },
  {
    name: "RF Detected",
    version: "ML v2.2.0",
    confidence: 95.0,
    accuracy: 94.0,
    description: "Detects and characterizes RF emissions",
    features: ["Frequency Spectrum", "Signal Strength", "Modulation"],
    ruleSet: "RF Detection Algorithm",
    passCriteria: "Authorized emissions only"
  },
  {
    name: "Subsatellite Deployment",
    version: "ML v1.7.1",
    confidence: 96.0,
    accuracy: 95.0,
    description: "Detects deployment of sub-satellites",
    features: ["Mass Changes", "Separation Events", "Multi-object Tracking"],
    ruleSet: "Deployment Detection",
    passCriteria: "No unauthorized deployments"
  },
  {
    name: "Optical Signature",
            version: "ML v2.6.0",
    confidence: 91.0,
    accuracy: 90.0,
    description: "Analyzes optical characteristics and changes",
    features: ["Brightness", "Spectral Data", "Temporal Variations"],
    ruleSet: "Optical Analysis",
    passCriteria: "Consistent with known profile"
  },
  {
    name: "Radar Signature",
    version: "ML v2.1.0",
    confidence: 94.0,
    accuracy: 93.0,
    description: "Analyzes radar cross-section and characteristics",
    features: ["RCS Measurements", "Doppler Shift", "Polarization"],
    ruleSet: "Radar Profile Analysis",
    passCriteria: "Matches expected signature"
  },
  {
    name: "Area Mass Ratio",
    version: "ML v1.9.0",
    confidence: 92.0,
    accuracy: 91.0,
    description: "Calculates and monitors area-to-mass ratio",
    features: ["Drag Coefficient", "Solar Pressure", "Atmospheric Effects"],
    ruleSet: "AMR Calculation",
    passCriteria: "Within expected range"
  },
  {
    name: "Proximity Operations",
    version: "ML v3.2.0",
    confidence: 98.0,
    accuracy: 97.0,
    description: "Detects and classifies proximity operations",
    features: ["Relative Motion", "Approach Patterns", "Safety Zones"],
    ruleSet: "Proximity Detection",
    passCriteria: "Safe separation maintained"
  },
  {
    name: "Tracking Anomalies",
    version: "ML v2.0.0",
    confidence: 93.0,
    accuracy: 92.0,
    description: "Identifies anomalies in tracking data",
    features: ["Data Gaps", "Measurement Errors", "Inconsistencies"],
    ruleSet: "Anomaly Detection",
    passCriteria: "No significant anomalies"
  },
  {
    name: "Imaging Maneuvers",
    version: "ML v2.5.0",
    confidence: 94.0,
    accuracy: 93.0,
    description: "Detects maneuvers resulting in valid remote-sensing passes",
    features: ["Ground Track", "Sensor Geometry", "Target Coverage"],
    ruleSet: "Imaging Opportunity Analysis",
    passCriteria: "Non-threatening imaging only"
  }
]

const ruleIndicators = [
  {
    name: "ITU/FCC Violation",
    version: "Rule v1.9.0",
    status: "Pass",
    description: "Checks compliance with ITU and FCC filings",
    ruleSet: "ITU/FCC Compliance Rules",
    passCriteria: "All RF activities comply with filings",
    features: ["RF Parameters", "Filing Data", "Compliance History"]
  },
  {
    name: "Analyst Consensus",
    version: "Rule v2.3.0",
    status: "Pass",
    description: "Aggregates analyst assessments",
    ruleSet: "Consensus Algorithm",
    passCriteria: "Majority agreement on assessment",
    features: ["Analyst Inputs", "Confidence Weights", "Historical Accuracy"]
  },
  {
    name: "System Response",
    version: "Rule v1.5.0",
    status: "Pass",
    description: "Monitors system response to stimulation",
    ruleSet: "Response Analysis",
    passCriteria: "Predictable responses only",
    features: ["Stimulus Type", "Response Time", "Response Pattern"]
  },
  {
    name: "Launch Site",
    version: "Rule v1.2.1",
    status: "Warning",
    description: "Identifies if object came from known threat launch site",
    ruleSet: "Launch Site Database",
    passCriteria: "Not from threat sites",
    features: ["Launch Location", "Vehicle Type", "Historical Data"]
  },
  {
    name: "UN Registry",
    version: "Rule v2.1.0",
    status: "Pass",
    description: "Checks if object is in UN satellite registry",
    ruleSet: "UN Registration Check",
    passCriteria: "Object is registered",
    features: ["Registration Status", "Owner Information", "Purpose"]
  },
  {
    name: "Camouflage Detection",
    version: "Rule v1.8.0",
    status: "Pass",
    description: "Detects attempts at concealment or deception",
    ruleSet: "Deception Detection",
    passCriteria: "No deception detected",
    features: ["Multi-spectral Analysis", "Behavioral Consistency", "Declaration Matching"]
  },
  {
    name: "Intent Assessment",
    version: "Rule v3.0.0",
    status: "Pass",
    description: "Assesses hostile or benign intent",
    ruleSet: "Intent Analysis Framework",
    passCriteria: "Benign intent confirmed",
    features: ["Behavioral Indicators", "Capability Assessment", "Historical Context"]
  }
]

const thresholdIndicators = [
  {
    name: "High Radiation Orbit",
    threshold: "1000 rad/day",
    current: 850,
    unit: "rad/day",
    status: "Pass",
    description: "Monitors if object is in high radiation environment"
  }
]

interface IndicatorCardProps {
  children: React.ReactNode
  expanded: boolean
  onToggle: () => void
}

function IndicatorCard({ children, expanded, onToggle }: IndicatorCardProps) {
  return (
    <Collapsible open={expanded}>
      <Card className="cursor-pointer" onClick={onToggle}>
        <CardHeader className="pb-3">
          {children}
        </CardHeader>
      </Card>
    </Collapsible>
  )
}

export default function IndicatorsPage() {
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set())

  const toggleCard = (cardName: string) => {
    const newExpanded = new Set(expandedCards)
    if (newExpanded.has(cardName)) {
      newExpanded.delete(cardName)
    } else {
      newExpanded.add(cardName)
    }
    setExpandedCards(newExpanded)
  }

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">CCDM Indicators</h2>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Indicators</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">19</div>
            <p className="text-xs text-muted-foreground">All 19 CCDM indicators</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Indicators</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">Machine learning powered</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rule Indicators</CardTitle>
            <Scale className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">6</div>
            <p className="text-xs text-muted-foreground">Rule-based logic</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Threshold Indicators</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1</div>
            <p className="text-xs text-muted-foreground">Threshold monitoring</p>
          </CardContent>
        </Card>
      </div>

      {/* Indicators Tabs */}
      <Tabs defaultValue="ml" className="space-y-4">
        <TabsList>
          <TabsTrigger value="ml">ML Indicators</TabsTrigger>
          <TabsTrigger value="rule">Rule Indicators</TabsTrigger>
          <TabsTrigger value="threshold">Threshold Indicators</TabsTrigger>
        </TabsList>

        <TabsContent value="ml" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mlIndicators.map((indicator) => (
              <IndicatorCard
                key={indicator.name}
                expanded={expandedCards.has(indicator.name)}
                onToggle={() => toggleCard(indicator.name)}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{indicator.name}</CardTitle>
                    <Badge variant="secondary" className="text-xs">
                      {indicator.version}
                    </Badge>
                  </div>
                  <CardDescription className="text-sm">
                    {indicator.description}
                  </CardDescription>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Confidence Score</span>
                      <span className="font-medium">{indicator.confidence}%</span>
                    </div>
                    <Progress value={indicator.confidence} className="h-2" />
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Training Accuracy</span>
                      <span className="font-medium">{indicator.accuracy}%</span>
                    </div>
                  </div>
                  <CollapsibleContent>
                    <div className="pt-4 space-y-2 border-t">
                      <div>
                        <p className="text-sm font-medium mb-1">Rule Set</p>
                        <p className="text-sm text-muted-foreground">{indicator.ruleSet}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-1">Pass Criteria</p>
                        <p className="text-sm text-muted-foreground">{indicator.passCriteria}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-1">Features</p>
                        <div className="flex flex-wrap gap-1">
                          {indicator.features.map((feature) => (
                            <Badge key={feature} variant="outline" className="text-xs">
                              {feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CollapsibleContent>
                  <div className="flex items-center justify-end">
                    {expandedCards.has(indicator.name) ? (
                      <ChevronUp className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                </div>
              </IndicatorCard>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="rule" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {ruleIndicators.map((indicator) => (
              <IndicatorCard
                key={indicator.name}
                expanded={expandedCards.has(indicator.name)}
                onToggle={() => toggleCard(indicator.name)}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{indicator.name}</CardTitle>
                    <Badge variant="secondary" className="text-xs">
                      {indicator.version}
                    </Badge>
                  </div>
                  <CardDescription className="text-sm">
                    {indicator.description}
                  </CardDescription>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Status</span>
                    <Badge
                      variant={indicator.status === "Pass" ? "default" : "destructive"}
                      className={indicator.status === "Warning" ? "bg-yellow-500 text-white" : ""}
                    >
                      {indicator.status}
                    </Badge>
                  </div>
                  <CollapsibleContent>
                    <div className="pt-4 space-y-2 border-t">
                      <div>
                        <p className="text-sm font-medium mb-1">Rule Set</p>
                        <p className="text-sm text-muted-foreground">{indicator.ruleSet}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-1">Pass Criteria</p>
                        <p className="text-sm text-muted-foreground">{indicator.passCriteria}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-1">Features</p>
                        <div className="flex flex-wrap gap-1">
                          {indicator.features.map((feature) => (
                            <Badge key={feature} variant="outline" className="text-xs">
                              {feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CollapsibleContent>
                  <div className="flex items-center justify-end">
                    {expandedCards.has(indicator.name) ? (
                      <ChevronUp className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                </div>
              </IndicatorCard>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="threshold" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {thresholdIndicators.map((indicator) => (
              <Card key={indicator.name}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{indicator.name}</CardTitle>
                    <Badge
                      variant={indicator.status === "Pass" ? "default" : "destructive"}
                    >
                      {indicator.status}
                    </Badge>
                  </div>
                  <CardDescription className="text-sm">
                    {indicator.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Threshold</span>
                      <span className="font-medium">{indicator.threshold}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Current Value</span>
                      <span className="font-medium">{indicator.current} {indicator.unit}</span>
                    </div>
                    <Progress 
                      value={(indicator.current / 1000) * 100} 
                      className="h-2"
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

