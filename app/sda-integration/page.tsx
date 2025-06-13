"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { RefreshCw, CheckCircle, AlertCircle, XCircle } from "lucide-react"

interface SubsystemStatus {
  id: string
  name: string
  status: "operational" | "degraded" | "offline" | "not_implemented"
  lastUpdate: string
  metrics?: {
    eventsProcessed?: number
    errorRate?: number
    latency?: number
  }
}

export default function SDAIntegrationPage() {
  const [systemStatus, setSystemStatus] = useState<any>(null)
  const [subsystemStatuses, setSubsystemStatuses] = useState<SubsystemStatus[]>([])
  const [loading, setLoading] = useState(true)

  const subsystems = [
    {
      id: "SS0",
      name: "Data Ingestion & Sensors",
      description: "All observational data and sensor inputs into Welders Arc",
      requirements: [
        "UDL integration for sensor data",
        "Automated collect request/response handling", 
        "Multi-phenomenology sensor fusion",
        "Real-time heartbeat monitoring"
      ],
      status: "Needs Implementation",
      priority: "Critical",
      implementation: 20
    },
    {
      id: "SS1", 
      name: "Target Modeling",
      description: "Dynamic target model database with RSO characteristics",
      requirements: [
        "Central repository for vehicle capabilities",
        "Kafka request/reply for missing data",
        "Integration with open source databases",
        "Real-time target updates"
      ],
      status: "Partially Implemented",
      priority: "High",
      implementation: 40
    },
    {
      id: "SS2",
      name: "State Estimation", 
      description: "UCT processing and orbit determination",
      requirements: [
        "Ensemble aggregator for multiple processors",
        "Real-time state updates from observations",
        "Automated UCT processing workflows",
        "Cross-catalog correlation"
      ],
      status: "Implemented", 
      priority: "Critical",
      implementation: 80
    },
    {
      id: "SS3",
      name: "Command & Control",
      description: "Sensor orchestration and collection management",
      requirements: [
        "Automated sensor scheduling",
        "Surveillance/custody/characterization missions",
        "Threat warning assessment",
        "PES (Payload Engagement Zone) analysis"
      ],
      status: "Basic Implementation",
      priority: "High",
      implementation: 30
    },
    {
      id: "SS4",
      name: "CCDM Evaluation",
      description: "Camouflage, Concealment, Deception & Maneuvering detection",
      requirements: [
        "19 automated indicators per Problem 16",
        "Node-RED workflow integration",
        "Object of Interest list generation",
        "Real-time anomaly scoring"
      ],
      status: "Good Foundation",
      priority: "Medium",
      implementation: 70
    },
    {
      id: "SS5",
      name: "Hostility Monitoring",
      description: "Intent assessment and threat evaluation",
      requirements: [
        "Weapon Engagement Zone prediction", 
        "Pattern of Life violation detection",
        "Pursuit maneuver identification",
        "Multi-phenomenology threat confirmation"
      ],
      status: "Implemented",
      priority: "Critical",
      implementation: 85
    },
    {
      id: "SS6",
      name: "Response Coordination",
      description: "Defensive course of action recommendations",
      requirements: [
        "Automated CoA generation",
        "Asset protection prioritization",
        "Mitigation strategy selection",
        "Real-time operator alerts"
      ],
      status: "Needs Implementation",
      priority: "High",
      implementation: 10
    }
  ]

  const eventProcessing = [
    "Launch Detection & Analysis",
    "Re-entry Tracking", 
    "Maneuver Detection",
    "Proximity Operations",
    "Separation Events",
    "Attitude Changes",
    "Link State Changes"
  ]

  const integrationNeeds = [
    {
      component: "Kafka Message Bus",
      description: "Event-driven architecture with 122+ message schemas",
      status: "Implemented",
      action: "Configure production Kafka cluster"
    },
    {
      component: "UDL Connectivity", 
      description: "Unified Data Library for sensor data access",
      status: "Implemented",
      action: "Obtain production UDL credentials"
    },
    {
      component: "Node-RED Workflows",
      description: "Visual workflow orchestration for CCDM",
      status: "Implemented", 
      action: "Deploy production Node-RED instance"
    },
    {
      component: "Global Data Marketplace",
      description: "Government contracting and subscription system",
      status: "Missing",
      action: "Register and integrate GDM APIs"
    }
  ]

  useEffect(() => {
    fetchSystemStatus()
    const interval = setInterval(fetchSystemStatus, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/v1/sda/status')
      if (response.ok) {
        const data = await response.json()
        setSystemStatus(data)
        updateSubsystemStatuses(data)
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    } finally {
      setLoading(false)
    }
  }

  const updateSubsystemStatuses = (data: any) => {
    // Map backend status to subsystem statuses
    const statuses: SubsystemStatus[] = [
      {
        id: "kafka",
        name: "Kafka Message Bus",
        status: data.details?.kafka_connected ? "operational" : "offline",
        lastUpdate: new Date().toISOString()
      },
      {
        id: "udl",
        name: "UDL Connection",
        status: data.details?.udl_connected ? "operational" : "offline",
        lastUpdate: new Date().toISOString()
      },
      {
        id: "node-red",
        name: "Node-RED Workflows",
        status: data.details?.node_red_connected ? "operational" : "offline",
        lastUpdate: new Date().toISOString()
      }
    ]
    setSubsystemStatuses(statuses)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "operational":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "degraded":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      case "offline":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />
    }
  }

  const getOverallProgress = () => {
    const total = subsystems.reduce((sum, s) => sum + s.implementation, 0)
    return Math.round(total / subsystems.length)
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">SDA Welders Arc Integration</h1>
        <div className="flex gap-2">
          <Badge variant="outline">Based on Tap Lab Requirements</Badge>
          <Button 
            size="sm" 
            variant="outline"
            onClick={fetchSystemStatus}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overall Progress */}
      <Card>
        <CardHeader>
          <CardTitle>Overall Implementation Progress</CardTitle>
          <CardDescription>
            Progress towards full SDA Welders Arc compliance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Total Progress</span>
              <span className="font-semibold">{getOverallProgress()}%</span>
            </div>
            <Progress value={getOverallProgress()} className="h-3" />
          </div>
          
          {/* Real-time Status */}
          <div className="grid grid-cols-3 gap-4 mt-6">
            {subsystemStatuses.map((status) => (
              <div key={status.id} className="flex items-center gap-2 p-3 border rounded">
                {getStatusIcon(status.status)}
                <div className="flex-1">
                  <p className="text-sm font-medium">{status.name}</p>
                  <p className="text-xs text-muted-foreground">{status.status}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Alert>
        <AlertDescription>
          This integration implements the event-driven, automated battle management system required by SDA. 
          AstroShield has been architecturally updated to support Kafka messaging, UDL data access, and Node-RED workflows.
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>Welders Arc Subsystem Implementation</CardTitle>
          <CardDescription>
            Seven subsystems with implementation status and requirements
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {subsystems.map((subsystem) => (
            <div key={subsystem.id} className="border rounded p-4">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <h4 className="font-semibold">{subsystem.id}: {subsystem.name}</h4>
                  <p className="text-sm text-muted-foreground">{subsystem.description}</p>
                </div>
                <div className="flex gap-2">
                  <Badge variant={subsystem.status === "Implemented" ? "default" : 
                                 subsystem.status === "Good Foundation" ? "default" :
                                 subsystem.status === "Partially Implemented" ? "secondary" :
                                 subsystem.status === "Basic Implementation" ? "secondary" : "destructive"}>
                    {subsystem.status}
                  </Badge>
                  <Badge variant={subsystem.priority === "Critical" ? "destructive" : 
                                 subsystem.priority === "High" ? "secondary" : "outline"}>
                    {subsystem.priority}
                  </Badge>
                </div>
              </div>
              
              {/* Implementation Progress */}
              <div className="mb-3">
                <div className="flex justify-between text-sm mb-1">
                  <span>Implementation</span>
                  <span>{subsystem.implementation}%</span>
                </div>
                <Progress value={subsystem.implementation} className="h-2" />
              </div>
              
              <div className="mt-3">
                <h5 className="text-sm font-medium mb-1">Requirements:</h5>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {subsystem.requirements.map((req, idx) => (
                    <li key={idx} className="flex items-center gap-2">
                      <span className="w-1 h-1 bg-muted-foreground rounded-full"></span>
                      {req}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Event Processing Workflows</CardTitle>
            <CardDescription>
              Seven automated event types implemented
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {eventProcessing.map((event, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 border rounded">
                  <span className="text-sm">{event}</span>
                  <Badge variant="default" className="text-xs">Implemented</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Integration Components</CardTitle>
            <CardDescription>
              Core infrastructure status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {integrationNeeds.map((need, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <h4 className="font-medium text-sm">{need.component}</h4>
                    <Badge 
                      variant={need.status === "Implemented" ? "default" : "destructive"} 
                      className="text-xs"
                    >
                      {need.status}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{need.description}</p>
                  <p className="text-xs font-medium text-blue-600">{need.action}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Implementation Roadmap</CardTitle>
          <CardDescription>
            Remaining work for full SDA compliance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="border-l-4 border-green-500 pl-4">
              <h4 className="font-semibold text-green-700">Completed: Core Infrastructure</h4>
              <ul className="text-sm space-y-1 mt-2">
                <li>✓ Kafka message bus architecture</li>
                <li>✓ UDL authentication and data access</li>
                <li>✓ UCT processing pipeline</li>
                <li>✓ WEZ prediction algorithms</li>
                <li>✓ Node-RED workflow backend</li>
              </ul>
            </div>
            
            <div className="border-l-4 border-yellow-500 pl-4">
              <h4 className="font-semibold text-yellow-700">In Progress: System Integration</h4>
              <ul className="text-sm space-y-1 mt-2">
                <li>• Complete sensor orchestration (SS3)</li>
                <li>• Finalize target model database (SS1)</li>
                <li>• Production deployment configuration</li>
                <li>• Performance optimization</li>
              </ul>
            </div>
            
            <div className="border-l-4 border-red-500 pl-4">
              <h4 className="font-semibold text-red-700">Remaining: Advanced Features</h4>
              <ul className="text-sm space-y-1 mt-2">
                <li>• Response coordination subsystem (SS6)</li>
                <li>• Global Data Marketplace integration</li>
                <li>• Classified system interfaces</li>
                <li>• Full Common Operating Picture UI</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      <Alert>
        <AlertDescription className="text-sm">
          <strong>Key Metrics for October 2025:</strong>
          <br />• Fully automated 100-day prototype-to-ops pipeline: <strong className="text-green-600">85% Complete</strong>
          <br />• Real-time UCT to resolved image capability: <strong className="text-green-600">Implemented</strong>
          <br />• Event-driven battle management: <strong className="text-green-600">Operational</strong>
          <br />• Integration with classified systems: <strong className="text-red-600">Pending</strong>
        </AlertDescription>
      </Alert>
    </div>
  )
} 