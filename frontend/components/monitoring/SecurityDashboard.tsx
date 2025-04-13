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
        const response = await getSecurityMetrics()
        if (!response.data) {
          setError(response.error?.message || 'No security data available')
          setMetrics(null)
          return
        }
        
        const current = response.data.security.current[0]
        setMetrics(current || null)
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
    return <div className="min-h-screen p-6">Loading security metrics...</div>
  }

  if (error) {
    return (
      <div className="min-h-screen p-6">
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="min-h-screen p-6">
        <Alert>
          <AlertTitle>No Data</AlertTitle>
          <AlertDescription>No security metrics available</AlertDescription>
        </Alert>
      </div>
    )
  }

  const securityScore = calculateSecurityScore(metrics)
  const threatLevel = getThreatLevel(metrics)

  return (
    <div className="min-h-screen p-6 pl-8">
      <div className="mb-8">
        <h2 className="text-2xl font-semibold tracking-tight">Security Overview</h2>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Security Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Progress value={securityScore} className="w-full h-2" />
              <p className="text-sm text-muted-foreground">
                {securityScore}% Secure
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Threat Level</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className={`text-2xl font-bold ${getThreatLevelColor(threatLevel)}`}>
                {threatLevel}
              </div>
              <p className="text-sm text-muted-foreground">
                Based on current metrics
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">HTTPS Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Progress value={metrics.httpsPercentage} className="w-full h-2" />
              <p className="text-sm text-muted-foreground">
                {metrics.httpsPercentage}% HTTPS Traffic
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Security Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-2xl font-bold">{metrics.cspViolations}</p>
                  <p className="text-sm text-muted-foreground">CSP Violations</p>
                </div>
                <div>
                  <p className="text-2xl font-bold">{metrics.blockedRequests}</p>
                  <p className="text-sm text-muted-foreground">Blocked Requests</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Error Handling</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-2xl font-bold">{metrics.sanitizedErrors}</p>
                  <p className="text-sm text-muted-foreground">Sanitized Errors</p>
                </div>
                <div>
                  <p className="text-2xl font-bold">{metrics.rateLimited}</p>
                  <p className="text-sm text-muted-foreground">Rate Limited</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-sm">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Last Updated</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {new Date(metrics.timestamp).toLocaleString()}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
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