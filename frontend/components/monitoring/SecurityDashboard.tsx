import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { getSecurityMetrics } from '@/lib/api-client'
import { LineChart } from '@/components/ui/charts'

interface SecurityMetricsState {
  httpsPercentage: number
  cspViolations: number
  blockedRequests: number
  rateLimited: number
  sanitizedErrors: number
  potentialLeaks: number
  timestamp: string
}

export function SecurityDashboard() {
  const [metrics, setMetrics] = useState<SecurityMetricsState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getSecurityMetrics()
        setMetrics(data.security.current[0] || null)
        setError(null)
      } catch (err) {
        setError('Failed to fetch security metrics')
        console.error('Security metrics error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Update every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <div>Loading security metrics...</div>
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )
  }

  if (!metrics) {
    return (
      <Alert>
        <AlertTitle>No Data</AlertTitle>
        <AlertDescription>No security metrics available</AlertDescription>
      </Alert>
    )
  }

  const securityScore = calculateSecurityScore(metrics)
  const threatLevel = getThreatLevel(metrics)

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Security Score</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Progress value={securityScore} className="w-full" />
            <p className="text-sm text-muted-foreground">
              {securityScore}% Secure
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Threat Level</CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getThreatLevelColor(threatLevel)}`}>
            {threatLevel}
          </div>
          <p className="text-sm text-muted-foreground">
            Based on current metrics
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>HTTPS Usage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {metrics.httpsPercentage}%
          </div>
          <Progress 
            value={metrics.httpsPercentage} 
            className="w-full mt-2"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Security Violations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>CSP Violations</span>
              <span className="font-bold">{metrics.cspViolations}</span>
            </div>
            <div className="flex justify-between">
              <span>Blocked Requests</span>
              <span className="font-bold">{metrics.blockedRequests}</span>
            </div>
            <div className="flex justify-between">
              <span>Rate Limited</span>
              <span className="font-bold">{metrics.rateLimited}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Error Handling</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>Sanitized Errors</span>
              <span className="font-bold">{metrics.sanitizedErrors}</span>
            </div>
            <div className="flex justify-between">
              <span>Potential Leaks</span>
              <span className="font-bold text-red-500">
                {metrics.potentialLeaks}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Last Updated</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            {new Date(metrics.timestamp).toLocaleString()}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function calculateSecurityScore(metrics: SecurityMetricsState): number {
  const weights = {
    https: 0.3,
    csp: 0.2,
    blocked: 0.2,
    rateLimit: 0.1,
    errors: 0.1,
    leaks: 0.1
  }

  const scores = {
    https: metrics.httpsPercentage,
    csp: Math.max(0, 100 - metrics.cspViolations * 10),
    blocked: Math.max(0, 100 - metrics.blockedRequests * 5),
    rateLimit: Math.max(0, 100 - metrics.rateLimited * 5),
    errors: Math.max(0, 100 - metrics.sanitizedErrors * 2),
    leaks: Math.max(0, 100 - metrics.potentialLeaks * 20)
  }

  return Math.round(
    Object.entries(weights).reduce(
      (score, [key, weight]) => score + scores[key as keyof typeof scores] * weight,
      0
    )
  )
}

function getThreatLevel(metrics: SecurityMetricsState): string {
  const score = calculateSecurityScore(metrics)
  if (score >= 90) return 'LOW'
  if (score >= 70) return 'MODERATE'
  if (score >= 50) return 'ELEVATED'
  if (score >= 30) return 'HIGH'
  return 'CRITICAL'
}

function getThreatLevelColor(level: string): string {
  switch (level) {
    case 'LOW': return 'text-green-500'
    case 'MODERATE': return 'text-yellow-500'
    case 'ELEVATED': return 'text-orange-500'
    case 'HIGH': return 'text-red-500'
    case 'CRITICAL': return 'text-red-700'
    default: return 'text-gray-500'
  }
} 