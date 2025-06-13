'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Play,
  Pause,
  CheckCircle,
  AlertTriangle,
  Clock,
  Activity,
  ArrowRight,
  Shield,
  Target,
  Zap,
  RefreshCw,
  Settings
} from 'lucide-react'

interface WorkflowStep {
  id: string
  name: string
  status: 'pending' | 'active' | 'completed' | 'failed'
  startTime?: Date
  endTime?: Date
  duration?: number
  confidence: number
  actions: string[]
  alerts?: string[]
}

interface OperationalWorkflow {
  id: string
  name: string
  type: 'maneuver' | 'proximity' | 'threat' | 'deployment'
  priority: 'low' | 'medium' | 'high' | 'critical'
  status: 'active' | 'paused' | 'completed'
  steps: WorkflowStep[]
  expertValidation: string
  automationLevel: number // 0-100%
}

export default function OperationalWorkflows() {
  const [workflows, setWorkflows] = useState<OperationalWorkflow[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null)
  const [isAutomationEnabled, setIsAutomationEnabled] = useState(true)

  // Tom Johnson's maneuver workflow template
  const maneuverWorkflowTemplate: WorkflowStep[] = [
    {
      id: 'detect',
      name: 'Maneuver Detection',
      status: 'pending',
      confidence: 0,
      actions: [
        'Monitor trajectory deviation',
        'Analyze burn signatures',
        'Calculate delta-V'
      ]
    },
    {
      id: 'start',
      name: 'Maneuver Start Confirmation',
      status: 'pending',
      confidence: 0,
      actions: [
        'Confirm thrust initiation',
        'Identify maneuver type',
        'Alert operators'
      ]
    },
    {
      id: 'ongoing',
      name: 'Maneuver Monitoring',
      status: 'pending',
      confidence: 0,
      actions: [
        'Track burn duration',
        'Monitor trajectory changes',
        'Update predictions'
      ]
    },
    {
      id: 'stop',
      name: 'Maneuver Completion',
      status: 'pending',
      confidence: 0,
      actions: [
        'Confirm thrust termination',
        'Calculate final orbit',
        'Update catalog'
      ]
    },
    {
      id: 'assessment',
      name: 'Post-Maneuver Assessment',
      status: 'pending',
      confidence: 0,
      actions: [
        'Evaluate maneuver intent',
        'Update threat assessment',
        'Generate report'
      ]
    }
  ]

  // Initialize workflows
  useEffect(() => {
    const initialWorkflows: OperationalWorkflow[] = [
      {
        id: 'wf-001',
        name: 'GEO-47291 Linear Drift Analysis',
        type: 'maneuver',
        priority: 'high',
        status: 'active',
        steps: maneuverWorkflowTemplate.map((step, index) => ({
          ...step,
          status: index === 0 ? 'completed' : index === 1 ? 'active' : 'pending',
          confidence: index === 0 ? 94.2 : index === 1 ? 67.8 : 0,
          startTime: index === 0 ? new Date(Date.now() - 3600000) : undefined,
          endTime: index === 0 ? new Date(Date.now() - 1800000) : undefined
        })),
        expertValidation: 'Tom Johnson',
        automationLevel: 75
      },
      {
        id: 'wf-002',
        name: 'LEO Proximity Alert Response',
        type: 'proximity',
        priority: 'critical',
        status: 'active',
        steps: [
          {
            id: 'detection',
            name: 'Proximity Detection',
            status: 'completed',
            confidence: 98.5,
            actions: ['Calculate miss distance', 'Assess collision risk'],
            startTime: new Date(Date.now() - 1200000),
            endTime: new Date(Date.now() - 900000)
          },
          {
            id: 'analysis',
            name: 'Conjunction Analysis',
            status: 'active',
            confidence: 82.3,
            actions: ['Refine predictions', 'Calculate probability'],
            startTime: new Date(Date.now() - 900000)
          },
          {
            id: 'decision',
            name: 'Maneuver Decision',
            status: 'pending',
            confidence: 0,
            actions: ['Evaluate options', 'Select maneuver strategy']
          },
          {
            id: 'execution',
            name: 'Execute Avoidance',
            status: 'pending',
            confidence: 0,
            actions: ['Command maneuver', 'Monitor execution']
          }
        ],
        expertValidation: 'Nathan Parrott',
        automationLevel: 60
      }
    ]

    setWorkflows(initialWorkflows)
    setSelectedWorkflow(initialWorkflows[0].id)

    // Simulate workflow progression
    const interval = setInterval(() => {
      setWorkflows(prev => prev.map(workflow => {
        if (workflow.status !== 'active' || !isAutomationEnabled) return workflow

        const activeStepIndex = workflow.steps.findIndex(s => s.status === 'active')
        if (activeStepIndex === -1) return workflow

        const activeStep = workflow.steps[activeStepIndex]
        const newConfidence = Math.min(activeStep.confidence + Math.random() * 5, 100)

        // Progress to next step if confidence is high enough
        if (newConfidence > 85 && activeStepIndex < workflow.steps.length - 1) {
          const updatedSteps = [...workflow.steps]
          updatedSteps[activeStepIndex] = {
            ...activeStep,
            status: 'completed',
            confidence: newConfidence,
            endTime: new Date()
          }
          updatedSteps[activeStepIndex + 1] = {
            ...updatedSteps[activeStepIndex + 1],
            status: 'active',
            startTime: new Date(),
            confidence: 20 + Math.random() * 20
          }

          return { ...workflow, steps: updatedSteps }
        }

        // Update confidence
        const updatedSteps = [...workflow.steps]
        updatedSteps[activeStepIndex] = {
          ...activeStep,
          confidence: newConfidence
        }

        return { ...workflow, steps: updatedSteps }
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [isAutomationEnabled])

  const getStatusIcon = (status: WorkflowStep['status']) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'active': return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />
      case 'failed': return <AlertTriangle className="h-5 w-5 text-red-500" />
      default: return <Clock className="h-5 w-5 text-gray-400" />
    }
  }

  const getPriorityVariant = (priority: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (priority) {
      case 'critical': return 'destructive'
      case 'high': return 'destructive'
      case 'medium': return 'secondary'
      default: return 'default'
    }
  }

  const selectedWorkflowData = workflows.find(w => w.id === selectedWorkflow)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Operational Workflows</h2>
          <p className="text-sm text-gray-600 mt-1">
            Tom Johnson's multi-event maneuver tracking with decision automation
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={isAutomationEnabled ? "default" : "outline"}
            size="sm"
            onClick={() => setIsAutomationEnabled(!isAutomationEnabled)}
          >
            {isAutomationEnabled ? (
              <>
                <Zap className="h-4 w-4 mr-1" />
                Automation ON
              </>
            ) : (
              <>
                <Settings className="h-4 w-4 mr-1" />
                Manual Mode
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Active Workflows */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {workflows.map(workflow => (
          <Card 
            key={workflow.id}
            className={`cursor-pointer transition-all ${
              selectedWorkflow === workflow.id ? 'ring-2 ring-blue-500' : ''
            }`}
            onClick={() => setSelectedWorkflow(workflow.id)}
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{workflow.name}</CardTitle>
                <Badge variant={getPriorityVariant(workflow.priority)}>
                  {workflow.priority.toUpperCase()}
                </Badge>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <span>Validated by {workflow.expertValidation}</span>
                <Badge variant="outline">{workflow.automationLevel}% Automated</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {/* Progress Overview */}
                <div className="flex items-center gap-2">
                  <Progress 
                    value={
                      (workflow.steps.filter(s => s.status === 'completed').length / 
                       workflow.steps.length) * 100
                    } 
                    className="flex-1"
                  />
                  <span className="text-sm font-medium">
                    {workflow.steps.filter(s => s.status === 'completed').length}/{workflow.steps.length}
                  </span>
                </div>

                {/* Current Step */}
                {workflow.steps.find(s => s.status === 'active') && (
                  <Alert>
                    <Activity className="h-4 w-4" />
                    <AlertTitle>Current Step</AlertTitle>
                    <AlertDescription>
                      {workflow.steps.find(s => s.status === 'active')?.name}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Detailed Workflow View */}
      {selectedWorkflowData && (
        <Card>
          <CardHeader>
            <CardTitle>Workflow Details: {selectedWorkflowData.name}</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="steps">
              <TabsList>
                <TabsTrigger value="steps">Workflow Steps</TabsTrigger>
                <TabsTrigger value="actions">Automated Actions</TabsTrigger>
                <TabsTrigger value="alerts">Alerts & Notifications</TabsTrigger>
              </TabsList>

              <TabsContent value="steps" className="space-y-4">
                {selectedWorkflowData.steps.map((step, index) => (
                  <div key={step.id} className="relative">
                    {index > 0 && (
                      <div className="absolute left-6 -top-4 h-4 w-0.5 bg-gray-300" />
                    )}
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0">
                        {getStatusIcon(step.status)}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">{step.name}</h4>
                          {step.status === 'active' && (
                            <Badge variant="default" className="animate-pulse">
                              In Progress
                            </Badge>
                          )}
                        </div>
                        
                        {step.confidence > 0 && (
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-600">Confidence:</span>
                            <Progress value={step.confidence} className="flex-1 max-w-xs" />
                            <span className="text-sm font-medium">
                              {step.confidence.toFixed(1)}%
                            </span>
                          </div>
                        )}

                        <div className="text-sm text-gray-600">
                          <p className="font-medium mb-1">Actions:</p>
                          <ul className="list-disc list-inside space-y-1">
                            {step.actions.map((action, i) => (
                              <li key={i}>{action}</li>
                            ))}
                          </ul>
                        </div>

                        {step.startTime && (
                          <div className="text-xs text-gray-500">
                            Started: {step.startTime.toLocaleTimeString()}
                            {step.endTime && (
                              <span> | Completed: {step.endTime.toLocaleTimeString()}</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </TabsContent>

              <TabsContent value="actions" className="space-y-4">
                <Alert>
                  <Shield className="h-4 w-4" />
                  <AlertTitle>Automated Actions Enabled</AlertTitle>
                  <AlertDescription>
                    The following actions will be executed automatically when confidence thresholds are met:
                  </AlertDescription>
                </Alert>

                <div className="space-y-3">
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium mb-2">Alert Generation</h4>
                    <ul className="text-sm space-y-1 text-gray-600">
                      <li>• Send notifications to operators when maneuver detected</li>
                      <li>• Update threat assessment dashboard in real-time</li>
                      <li>• Generate automated reports for command review</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium mb-2">Data Collection</h4>
                    <ul className="text-sm space-y-1 text-gray-600">
                      <li>• Automatically collect sensor data during events</li>
                      <li>• Archive trajectory information for analysis</li>
                      <li>• Update object catalog with new parameters</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium mb-2">Decision Support</h4>
                    <ul className="text-sm space-y-1 text-gray-600">
                      <li>• Calculate optimal response strategies</li>
                      <li>• Prepare maneuver recommendations</li>
                      <li>• Queue commands for operator approval</li>
                    </ul>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="alerts" className="space-y-4">
                <div className="space-y-3">
                  {selectedWorkflowData.priority === 'critical' && (
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Critical Priority Alert</AlertTitle>
                      <AlertDescription>
                        This workflow has triggered critical alerts requiring immediate attention
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Maneuver Detection Alert</span>
                      <Badge variant="default">Sent</Badge>
                    </div>
                    <p className="text-sm text-gray-600">
                      Notified: Space Operations Center, Mission Planning Team
                    </p>
                  </div>

                  <div className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Trajectory Update</span>
                      <Badge variant="outline">Pending</Badge>
                    </div>
                    <p className="text-sm text-gray-600">
                      Awaiting maneuver completion for catalog update
                    </p>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 