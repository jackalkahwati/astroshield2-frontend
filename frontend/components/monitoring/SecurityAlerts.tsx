import { useEffect, useState } from 'react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { getSecurityMetrics } from '@/lib/api-client'
import { Bell, Shield, AlertTriangle, XCircle } from 'lucide-react'

interface SecurityAlert {
  id: string
  type: 'csp' | 'blocked' | 'rateLimit' | 'leak'
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  timestamp: string
}

interface AlertState {
  alerts: SecurityAlert[]
  lastCheck: string
}

export function SecurityAlerts() {
  const [alertState, setAlertState] = useState<AlertState>({
    alerts: [],
    lastCheck: new Date().toISOString()
  })

  useEffect(() => {
    const checkAlerts = async () => {
      try {
        const response = await getSecurityMetrics()
        if (!response.data) {
          console.error(response.error?.message || "No security data available")
          return
        }
        
        const current = response.data.security.current[0]
        
        if (!current) return

        const newAlerts: SecurityAlert[] = []

        // Check for CSP violations
        if (current.cspViolations > 0) {
          newAlerts.push({
            id: `csp-${Date.now()}`,
            type: 'csp',
            severity: current.cspViolations > 10 ? 'high' : 'medium',
            message: `${current.cspViolations} CSP violations detected`,
            timestamp: current.timestamp
          })
        }

        // Check for blocked requests
        if (current.blockedRequests > 0) {
          newAlerts.push({
            id: `blocked-${Date.now()}`,
            type: 'blocked',
            severity: current.blockedRequests > 5 ? 'high' : 'medium',
            message: `${current.blockedRequests} suspicious requests blocked`,
            timestamp: current.timestamp
          })
        }

        // Check for rate limiting
        if (current.rateLimited > 0) {
          newAlerts.push({
            id: `rate-${Date.now()}`,
            type: 'rateLimit',
            severity: current.rateLimited > 20 ? 'critical' : 'high',
            message: `${current.rateLimited} requests rate limited`,
            timestamp: current.timestamp
          })
        }

        // Check for potential data leaks
        if (current.potentialLeaks > 0) {
          newAlerts.push({
            id: `leak-${Date.now()}`,
            type: 'leak',
            severity: 'critical',
            message: `${current.potentialLeaks} potential data leaks detected`,
            timestamp: current.timestamp
          })
        }

        setAlertState(prev => ({
          alerts: [...newAlerts, ...prev.alerts].slice(0, 10), // Keep last 10 alerts
          lastCheck: new Date().toISOString()
        }))
      } catch (error) {
        console.error('Failed to check security alerts:', error)
      }
    }

    checkAlerts()
    const interval = setInterval(checkAlerts, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  if (alertState.alerts.length === 0) {
    return (
      <Alert>
        <Shield className="h-4 w-4" />
        <AlertTitle>Security Status</AlertTitle>
        <AlertDescription>
          No active security alerts
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Security Alerts</h3>
        <Badge variant="outline">
          <Bell className="h-3 w-3 mr-1" />
          {alertState.alerts.length} Active
        </Badge>
      </div>

      <div className="space-y-2">
        {alertState.alerts.map(alert => (
          <Alert
            key={alert.id}
            variant={getAlertVariant(alert.severity)}
          >
            {getAlertIcon(alert.type)}
            <AlertTitle className="flex items-center gap-2">
              {getAlertTitle(alert.type)}
              <Badge variant={getAlertBadgeVariant(alert.severity)}>
                {alert.severity.toUpperCase()}
              </Badge>
            </AlertTitle>
            <AlertDescription className="mt-1">
              <p>{alert.message}</p>
              <p className="text-sm text-muted-foreground mt-1">
                {new Date(alert.timestamp).toLocaleString()}
              </p>
            </AlertDescription>
          </Alert>
        ))}
      </div>

      <p className="text-sm text-muted-foreground">
        Last checked: {new Date(alertState.lastCheck).toLocaleString()}
      </p>
    </div>
  )
}

function getAlertVariant(severity: SecurityAlert['severity']): "default" | "destructive" {
  switch (severity) {
    case 'critical':
    case 'high':
      return 'destructive'
    default:
      return 'default'
  }
}

function getAlertBadgeVariant(severity: SecurityAlert['severity']): "default" | "destructive" | "outline" {
  switch (severity) {
    case 'critical':
      return 'destructive'
    case 'high':
      return 'default'
    default:
      return 'outline'
  }
}

function getAlertTitle(type: SecurityAlert['type']): string {
  switch (type) {
    case 'csp':
      return 'Content Security Policy Violation'
    case 'blocked':
      return 'Suspicious Activity Blocked'
    case 'rateLimit':
      return 'Rate Limit Exceeded'
    case 'leak':
      return 'Potential Data Leak'
    default:
      return 'Security Alert'
  }
}

function getAlertIcon(type: SecurityAlert['type']) {
  switch (type) {
    case 'csp':
      return <Shield className="h-4 w-4" />
    case 'blocked':
      return <XCircle className="h-4 w-4" />
    case 'rateLimit':
      return <AlertTriangle className="h-4 w-4" />
    case 'leak':
      return <AlertTriangle className="h-4 w-4" />
  }
} 