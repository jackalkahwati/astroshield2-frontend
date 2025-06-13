"use client"

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Checkbox } from '@/components/ui/checkbox'
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Zap,
  Target,
  Shield,
  Navigation,
  Radio,
  Satellite,
  Users,
  MessageSquare,
  Play,
  Pause,
  SkipForward,
  ChevronRight,
  Info,
  TrendingUp,
  Activity,
  Gauge,
  FileText,
  Download,
  RefreshCw,
  Settings,
  AlertCircle,
  ArrowRight,
  Lightbulb
} from 'lucide-react'
import { format } from 'date-fns'
import { motion, AnimatePresence } from 'framer-motion'

interface ActionRecommendation {
  action_id: string
  action_type: string
  priority: string
  title: string
  description: string
  rationale: string[]
  target_subsystem: string
  implementation_steps: Array<{
    step: number
    action: string
    details: Record<string, any>
  }>
  resource_requirements: Record<string, any>
  confidence: number
  execution_time: string
  event_id: string
  constraints?: Record<string, any>
  preconditions?: string[]
  success_criteria?: string[]
  alternative_actions?: string[]
}

interface DecisionWorkflow {
  workflow_id: string
  name: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  current_step: number
  total_steps: number
  started_at: string
  estimated_completion: string
  actions: ActionRecommendation[]
}

interface ActionEffectiveness {
  action_type: string
  success_rate: number
  average_execution_time: number
  last_used: string
  total_uses: number
}

const priorityConfig = {
  immediate: { color: 'bg-red-500', icon: Zap, label: 'Immediate' },
  urgent: { color: 'bg-orange-500', icon: AlertTriangle, label: 'Urgent' },
  high: { color: 'bg-yellow-500', icon: TrendingUp, label: 'High' },
  moderate: { color: 'bg-blue-500', icon: Activity, label: 'Moderate' },
  low: { color: 'bg-gray-500', icon: Info, label: 'Low' },
  routine: { color: 'bg-green-500', icon: CheckCircle, label: 'Routine' }
}

const actionTypeIcons: Record<string, any> = {
  increase_tracking: Radio,
  request_additional_sensors: Satellite,
  execute_avoidance_maneuver: Navigation,
  emergency_maneuver: Zap,
  coordinate_with_operator: Users,
  notify_command: MessageSquare,
  alert_space_force: Shield,
  continue_monitoring: Activity,
  change_attitude: Target
}

// Utility function for safe date formatting
const formatDateSafe = (dateString: string | null | undefined, formatStr: string, fallback: string = 'Invalid time'): string => {
  try {
    if (!dateString) return fallback
    const date = new Date(dateString)
    if (isNaN(date.getTime())) return fallback
    return format(date, formatStr)
  } catch (error) {
    console.error('Date formatting error:', error)
    return fallback
  }
}

export function DecisionSupportDashboard() {
  // Mock data for demo
  const mockRecommendations: ActionRecommendation[] = [
    {
      action_id: 'ACT-20250609-001',
      action_type: 'emergency_maneuver',
      priority: 'immediate',
      title: 'Execute Emergency Avoidance Maneuver',
      description: 'Critical proximity event detected. Immediate maneuver required to avoid collision with debris.',
      rationale: [
        'Collision probability exceeds safety threshold (>0.0001)',
        'Time to closest approach: 47 minutes',
        'No alternative mitigation options available'
      ],
      target_subsystem: 'SS3',
      implementation_steps: [
        { step: 1, action: 'Calculate optimal maneuver vector', details: { delta_v: '2.1 m/s', burn_duration: '45s' } },
        { step: 2, action: 'Upload maneuver commands', details: { execution_time: '2025-01-23T11:15:00Z' } },
        { step: 3, action: 'Execute maneuver', details: { verification_required: true } }
      ],
      resource_requirements: { fuel_kg: 12.5, execution_time_minutes: 15, operator_approval: true },
      confidence: 0.95,
      execution_time: new Date(Date.now() + 900000).toISOString(),
      event_id: 'PROX-20250609-001',
      constraints: { fuel_limit: 'within_tolerance', maneuver_window: 'available' },
      success_criteria: ['Range > 5km at TCA', 'Maneuver executed within +/- 30s'],
      alternative_actions: ['coordinate_with_debris_owner', 'request_conjunction_update']
    },
    {
      action_id: 'ACT-20250609-002',
      action_type: 'increase_tracking',
      priority: 'urgent',
      title: 'Increase Tracking Frequency',
      description: 'Enhance tracking of high-risk object to improve conjunction prediction accuracy.',
      rationale: [
        'Object tracking uncertainty above normal levels',
        'Improved tracking will reduce prediction error by 40%',
        'Critical for maneuver planning decision'
      ],
      target_subsystem: 'SS0',
      implementation_steps: [
        { step: 1, action: 'Contact tracking stations', details: { stations: ['GEODSS', 'SBV'] } },
        { step: 2, action: 'Increase observation frequency', details: { new_frequency: 'every_2_hours' } }
      ],
      resource_requirements: { tracking_time_hours: 24, priority_level: 'high' },
      confidence: 0.88,
      execution_time: new Date(Date.now() + 300000).toISOString(),
      event_id: 'PROX-20250609-002'
    },
    {
      action_id: 'ACT-20250609-003',
      action_type: 'coordinate_with_operator',
      priority: 'high',
      title: 'Coordinate with Satellite Operator',
      description: 'Contact satellite operator to discuss potential coordinated maneuver options.',
      rationale: [
        'Both objects are maneuverable',
        'Coordinated maneuver more fuel-efficient',
        'Operator has indicated willingness to coordinate'
      ],
      target_subsystem: 'SS6',
      implementation_steps: [
        { step: 1, action: 'Initiate operator contact', details: { operator: 'SpaceX', contact_method: 'secure_channel' } },
        { step: 2, action: 'Share conjunction data', details: { data_classification: 'FOUO' } },
        { step: 3, action: 'Negotiate maneuver plan', details: { deadline: '2025-01-23T14:00:00Z' } }
      ],
      resource_requirements: { coordination_time_hours: 4, analyst_time: 2 },
      confidence: 0.75,
      execution_time: new Date(Date.now() + 1800000).toISOString(),
      event_id: 'PROX-20250609-002'
    }
  ]

  const mockWorkflows: DecisionWorkflow[] = [
    {
      workflow_id: 'WF-20250609-001',
      name: 'Emergency Collision Avoidance',
      status: 'in_progress',
      current_step: 2,
      total_steps: 4,
      started_at: new Date(Date.now() - 1800000).toISOString(),
      estimated_completion: new Date(Date.now() + 900000).toISOString(),
      actions: mockRecommendations.slice(0, 1)
    },
    {
      workflow_id: 'WF-20250609-002',
      name: 'Routine Conjunction Assessment',
      status: 'pending',
      current_step: 1,
      total_steps: 3,
      started_at: new Date(Date.now() - 900000).toISOString(),
      estimated_completion: new Date(Date.now() + 3600000).toISOString(),
      actions: mockRecommendations.slice(1)
    }
  ]

  const mockEffectiveness: Record<string, ActionEffectiveness> = {
    emergency_maneuver: { action_type: 'emergency_maneuver', success_rate: 0.94, average_execution_time: 18, last_used: '2025-01-20T08:30:00Z', total_uses: 12 },
    increase_tracking: { action_type: 'increase_tracking', success_rate: 0.98, average_execution_time: 45, last_used: '2025-01-22T14:15:00Z', total_uses: 156 },
    coordinate_with_operator: { action_type: 'coordinate_with_operator', success_rate: 0.67, average_execution_time: 240, last_used: '2025-01-21T11:45:00Z', total_uses: 34 }
  }

  const [recommendations, setRecommendations] = useState<ActionRecommendation[]>(mockRecommendations)
  const [workflows, setWorkflows] = useState<DecisionWorkflow[]>(mockWorkflows)
  const [selectedRecommendation, setSelectedRecommendation] = useState<ActionRecommendation | null>(null)
  const [effectiveness, setEffectiveness] = useState<Record<string, ActionEffectiveness>>(mockEffectiveness)
  const [activeTab, setActiveTab] = useState('recommendations')
  const [filterPriority, setFilterPriority] = useState<string>('all')
  const [autoExecute, setAutoExecute] = useState(false)
  const [executingActions, setExecutingActions] = useState<Set<string>>(new Set())
  const [isUsingSampleData, setIsUsingSampleData] = useState(true)

  // Fetch recommendations
  useEffect(() => {
    fetchRecommendations()
    fetchWorkflows()
    fetchEffectiveness()
    
    // Add fallback data if API returns empty after a delay
    setTimeout(() => {
      if (recommendations.length === 0) {
        setRecommendations(mockRecommendations)
      }
      if (workflows.length === 0) {
        setWorkflows(mockWorkflows)
      }
      if (Object.keys(effectiveness).length === 0) {
        setEffectiveness(mockEffectiveness)
      }
    }, 3000)
    
    // Auto-refresh
    const interval = setInterval(() => {
      fetchRecommendations()
      fetchWorkflows()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [filterPriority])

  const fetchRecommendations = async () => {
    try {
      const params = filterPriority !== 'all' ? `?priority=${filterPriority}` : ''
      const response = await fetch(`/api/v1/response-recommendations/active${params}`)
      if (response.ok) {
        const data = await response.json()
        // Use API data if available, otherwise fallback to mock data
        if (data.length > 0) {
          setRecommendations(data)
          setIsUsingSampleData(false)
        } else {
          setRecommendations(mockRecommendations)
          setIsUsingSampleData(true)
        }
      }
    } catch (error) {
      console.error('Failed to fetch recommendations:', error)
      // On error, use mock data as fallback
      setRecommendations(mockRecommendations)
    }
  }

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('/api/v1/decision-workflows/active')
      if (response.ok) {
        const data = await response.json()
        // Use API data if available, otherwise fallback to mock data
        setWorkflows(data.length > 0 ? data : mockWorkflows)
      }
    } catch (error) {
      console.error('Failed to fetch workflows:', error)
      // On error, use mock data as fallback
      setWorkflows(mockWorkflows)
    }
  }

  const fetchEffectiveness = async () => {
    try {
      const response = await fetch('/api/v1/response-recommendations/effectiveness')
      if (response.ok) {
        const data = await response.json()
        // Use API data if available, otherwise fallback to mock data
        setEffectiveness(Object.keys(data).length > 0 ? data : mockEffectiveness)
      }
    } catch (error) {
      console.error('Failed to fetch effectiveness:', error)
      // On error, use mock data as fallback
      setEffectiveness(mockEffectiveness)
    }
  }

  const executeAction = async (recommendation: ActionRecommendation) => {
    setExecutingActions(prev => new Set(prev).add(recommendation.action_id))
    
    try {
      const response = await fetch(`/api/v1/response-recommendations/${recommendation.action_id}/execute`, {
        method: 'POST'
      })
      
      if (response.ok) {
        // Refresh recommendations
        await fetchRecommendations()
        await fetchWorkflows()
      }
    } catch (error) {
      console.error('Failed to execute action:', error)
    } finally {
      setExecutingActions(prev => {
        const newSet = new Set(prev)
        newSet.delete(recommendation.action_id)
        return newSet
      })
    }
  }

  const dismissRecommendation = async (recommendationId: string) => {
    try {
      const response = await fetch(`/api/v1/response-recommendations/${recommendationId}/dismiss`, {
        method: 'POST'
      })
      
      if (response.ok) {
        setRecommendations(prev => prev.filter(r => r.action_id !== recommendationId))
      }
    } catch (error) {
      console.error('Failed to dismiss recommendation:', error)
    }
  }

  const getPriorityRecommendations = () => {
    return recommendations.filter(r => 
      r.priority === 'immediate' || r.priority === 'urgent'
    ).slice(0, 3)
  }

  const getTimeUntilExecution = (executionTime: string) => {
    try {
      if (!executionTime) return 'Time not set'
      
      const now = new Date()
      const execTime = new Date(executionTime)
      
      // Check if the date is valid
      if (isNaN(execTime.getTime())) return 'Invalid time'
      
      const diff = execTime.getTime() - now.getTime()
      
      if (diff < 0) return 'Overdue'
      if (diff < 60000) return 'Less than 1 minute'
      if (diff < 3600000) return `${Math.floor(diff / 60000)} minutes`
      if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours`
      return `${Math.floor(diff / 86400000)} days`
    } catch (error) {
      console.error('Error calculating time until execution:', error)
      return 'Time unavailable'
    }
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex justify-between items-center">
        {isUsingSampleData && (
          <Badge variant="outline" className="text-xs">
            Sample Data
          </Badge>
        )}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Checkbox
              checked={autoExecute}
              onCheckedChange={(checked) => setAutoExecute(checked as boolean)}
            />
            <span className="text-sm">Auto-execute low-risk actions</span>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchRecommendations()
              fetchWorkflows()
            }}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Priority Alerts */}
      {getPriorityRecommendations().length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Immediate Action Required</AlertTitle>
          <AlertDescription>
            {getPriorityRecommendations().length} high-priority recommendations require your attention
          </AlertDescription>
        </Alert>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {getPriorityRecommendations().map((rec) => {
          const config = priorityConfig[rec.priority as keyof typeof priorityConfig]
          const Icon = actionTypeIcons[rec.action_type] || Activity
          const isExecuting = executingActions.has(rec.action_id)
          
          return (
            <div key={rec.action_id}>
              <Card className="border-2 border-red-200 bg-red-50">
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <div className={`p-2 rounded-full ${config.color} text-white`}>
                        <config.icon className="h-4 w-4" />
                      </div>
                      <div>
                        <CardTitle className="text-base">{rec.title}</CardTitle>
                        <p className="text-xs text-gray-500 mt-1">
                          Execute by {formatDateSafe(rec.execution_time, 'HH:mm', 'Time not set')}
                        </p>
                      </div>
                    </div>
                    <Badge variant="destructive">{config.label}</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-3">{rec.description}</p>
                  
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      className="flex-1"
                      onClick={() => executeAction(rec)}
                      disabled={isExecuting}
                    >
                      {isExecuting ? (
                        <>
                          <Clock className="h-4 w-4 mr-1 animate-spin" />
                          Executing...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-1" />
                          Execute
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setSelectedRecommendation(rec)}
                    >
                      Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          )
        })}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="recommendations">
            <Lightbulb className="h-4 w-4 mr-2" />
            Recommendations
          </TabsTrigger>
          <TabsTrigger value="workflows">
            <Activity className="h-4 w-4 mr-2" />
            Workflows
          </TabsTrigger>
          <TabsTrigger value="implementation">
            <Settings className="h-4 w-4 mr-2" />
            Implementation
          </TabsTrigger>
          <TabsTrigger value="effectiveness">
            <Gauge className="h-4 w-4 mr-2" />
            Effectiveness
          </TabsTrigger>
        </TabsList>

        <TabsContent value="recommendations" className="space-y-4">
          {/* Filter */}
          <Card>
            <CardContent className="p-4">
              <div className="flex gap-2">
                {Object.entries(priorityConfig).map(([key, config]) => (
                  <Button
                    key={key}
                    variant={filterPriority === key ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setFilterPriority(key)}
                  >
                    <config.icon className="h-4 w-4 mr-1" />
                    {config.label}
                  </Button>
                ))}
                <Button
                  variant={filterPriority === 'all' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilterPriority('all')}
                >
                  All
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Recommendations List */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {recommendations.map((rec) => {
              const config = priorityConfig[rec.priority as keyof typeof priorityConfig]
              const Icon = actionTypeIcons[rec.action_type] || Activity
              const isExecuting = executingActions.has(rec.action_id)
              const effectivenessData = effectiveness[rec.action_type]
              
              return (
                <Card key={rec.action_id}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div className="flex items-center gap-2">
                        <Icon className="h-5 w-5 text-gray-600" />
                        <div>
                          <CardTitle className="text-base">{rec.title}</CardTitle>
                          <CardDescription className="text-xs">
                            {rec.target_subsystem} • {getTimeUntilExecution(rec.execution_time)}
                          </CardDescription>
                        </div>
                      </div>
                      <Badge className={config.color + ' text-white'}>
                        {config.label}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-gray-600">{rec.description}</p>
                    
                    {/* Confidence & Effectiveness */}
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <p className="text-xs text-gray-500">Confidence</p>
                        <div className="flex items-center gap-2">
                          <Progress value={rec.confidence * 100} className="h-2" />
                          <span className="text-xs">{(rec.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      {effectivenessData && (
                        <div>
                          <p className="text-xs text-gray-500">Historical Success</p>
                          <div className="flex items-center gap-2">
                            <Progress value={effectivenessData.success_rate * 100} className="h-2" />
                            <span className="text-xs">{(effectivenessData.success_rate * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {/* Rationale */}
                    <div>
                      <p className="text-xs font-medium text-gray-700 mb-1">Rationale:</p>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {rec.rationale.slice(0, 2).map((reason, idx) => (
                          <li key={idx} className="flex items-start gap-1">
                            <ChevronRight className="h-3 w-3 mt-0.5 flex-shrink-0" />
                            <span>{reason}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    {/* Constraints */}
                    {rec.constraints && Object.keys(rec.constraints).length > 0 && (
                      <Alert className="py-2">
                        <AlertCircle className="h-3 w-3" />
                        <AlertDescription className="text-xs">
                          Constraints: {Object.keys(rec.constraints).join(', ')}
                        </AlertDescription>
                      </Alert>
                    )}
                    
                    {/* Actions */}
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        className="flex-1"
                        onClick={() => executeAction(rec)}
                        disabled={isExecuting}
                      >
                        {isExecuting ? 'Executing...' : 'Execute'}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setSelectedRecommendation(rec)}
                      >
                        Details
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => dismissRecommendation(rec.action_id)}
                      >
                        Dismiss
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        <TabsContent value="workflows">
          <div className="space-y-4">
            {workflows.map((workflow) => (
              <Card key={workflow.workflow_id}>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>{workflow.name}</CardTitle>
                      <CardDescription>
                        Started {formatDateSafe(workflow.started_at, 'PPp', 'Unknown')}
                      </CardDescription>
                    </div>
                    <Badge
                      variant={
                        workflow.status === 'completed' ? 'default' :
                        workflow.status === 'failed' ? 'destructive' :
                        workflow.status === 'in_progress' ? 'secondary' :
                        'outline'
                      }
                    >
                      {workflow.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Progress */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                      <span>Step {workflow.current_step} of {workflow.total_steps}</span>
                      <span className="text-gray-500">
                        Est. completion: {formatDateSafe(workflow.estimated_completion, 'HH:mm', 'Unknown')}
                      </span>
                    </div>
                    <Progress 
                      value={(workflow.current_step / workflow.total_steps) * 100} 
                      className="h-2"
                    />
                  </div>
                  
                  {/* Actions in Workflow */}
                  <div className="space-y-2">
                    {workflow.actions.map((action, idx) => {
                      const Icon = actionTypeIcons[action.action_type] || Activity
                      const isCompleted = idx < workflow.current_step
                      const isCurrent = idx === workflow.current_step - 1
                      
                      return (
                        <div
                          key={action.action_id}
                          className={`flex items-center gap-3 p-2 rounded ${
                            isCurrent ? 'bg-blue-50 border border-blue-200' :
                            isCompleted ? 'opacity-60' : 'opacity-40'
                          }`}
                        >
                          <div className={`p-1.5 rounded-full ${
                            isCompleted ? 'bg-green-100 text-green-600' :
                            isCurrent ? 'bg-blue-100 text-blue-600' :
                            'bg-gray-100 text-gray-400'
                          }`}>
                            {isCompleted ? (
                              <CheckCircle className="h-4 w-4" />
                            ) : (
                              <Icon className="h-4 w-4" />
                            )}
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium">{action.title}</p>
                            <p className="text-xs text-gray-500">{action.target_subsystem}</p>
                          </div>
                          {isCurrent && (
                            <Badge variant="outline" className="text-xs">
                              In Progress
                            </Badge>
                          )}
                        </div>
                      )
                    })}
                  </div>
                  
                  {/* Workflow Actions */}
                  <div className="flex gap-2 mt-4">
                    {workflow.status === 'in_progress' && (
                      <>
                        <Button size="sm" variant="outline">
                          <Pause className="h-4 w-4 mr-1" />
                          Pause
                        </Button>
                        <Button size="sm" variant="outline">
                          <SkipForward className="h-4 w-4 mr-1" />
                          Skip Step
                        </Button>
                      </>
                    )}
                    <Button size="sm" variant="outline">
                      <FileText className="h-4 w-4 mr-1" />
                      View Log
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
            
            {workflows.length === 0 && (
              <Card>
                <CardContent className="text-center py-8">
                  <Activity className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                  <p className="text-gray-500">No active workflows</p>
                  <p className="text-sm text-gray-400">Workflows are created when executing complex multi-step actions</p>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="implementation">
          {selectedRecommendation && (
            <Card>
              <CardHeader>
                <CardTitle>Implementation Details: {selectedRecommendation.title}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Resource Requirements */}
                <div>
                  <h4 className="font-medium mb-2">Resource Requirements</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedRecommendation.resource_requirements).map(([key, value]) => (
                      <div key={key} className="flex justify-between p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-600">{key.replace(/_/g, ' ')}</span>
                        <span className="text-sm font-medium">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Preconditions */}
                {selectedRecommendation.preconditions && (
                  <div>
                    <h4 className="font-medium mb-2">Preconditions</h4>
                    <ul className="space-y-1">
                      {selectedRecommendation.preconditions.map((condition, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <CheckCircle className="h-4 w-4 text-gray-400 mt-0.5" />
                          <span>{condition}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Implementation Steps */}
                <div>
                  <h4 className="font-medium mb-2">Implementation Steps</h4>
                  <div className="space-y-3">
                    {selectedRecommendation.implementation_steps.map((step) => (
                      <div key={step.step} className="flex gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                          {step.step}
                        </div>
                        <div className="flex-1">
                          <p className="font-medium text-sm">{step.action}</p>
                          <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-auto">
                            {JSON.stringify(step.details, null, 2)}
                          </pre>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Success Criteria */}
                {selectedRecommendation.success_criteria && (
                  <div>
                    <h4 className="font-medium mb-2">Success Criteria</h4>
                    <ul className="space-y-1">
                      {selectedRecommendation.success_criteria.map((criteria, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <Target className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>{criteria}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Alternative Actions */}
                {selectedRecommendation.alternative_actions && selectedRecommendation.alternative_actions.length > 0 && (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>Alternative Actions</AlertTitle>
                    <AlertDescription>
                      {selectedRecommendation.alternative_actions.join(', ')}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="effectiveness">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(effectiveness).map(([actionType, data]) => {
              const Icon = actionTypeIcons[actionType] || Activity
              
              return (
                <Card key={actionType}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Icon className="h-5 w-5 text-gray-600" />
                      <CardTitle className="text-base">
                        {actionType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm text-gray-500">Success Rate</span>
                          <span className="text-sm font-medium">
                            {(data.success_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                        <Progress value={data.success_rate * 100} className="h-2" />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <p className="text-gray-500">Avg Time</p>
                          <p className="font-medium">{data.average_execution_time.toFixed(1)}s</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Total Uses</p>
                          <p className="font-medium">{data.total_uses}</p>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-500">
                                                    Last used: {formatDateSafe(data.last_used, 'PPp', 'Never')}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
} 