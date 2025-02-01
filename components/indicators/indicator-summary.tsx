import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Shield, AlertTriangle, Activity } from "lucide-react"

interface IndicatorsData {
  udl: {
    satellites: Array<{
      id: string;
      indicators: Record<string, any>;
    }>;
  };
  stability: {
    status: string;
    metrics: Record<string, any>;
  };
}

interface IndicatorSummaryProps {
  data: IndicatorsData | null;
}

export function IndicatorSummary({ data }: IndicatorSummaryProps) {
  const totalSatellites = data?.udl.satellites.length || 0
  const activeAlerts = data?.stability.metrics?.alerts?.length || 0
  const systemHealth = data?.stability.status === "stable" ? "98%" : "85%"

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Satellites</CardTitle>
          <Shield className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalSatellites}</div>
          <p className="text-xs text-muted-foreground">
            Active satellites being monitored
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
          <AlertTriangle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{activeAlerts}</div>
          <p className="text-xs text-muted-foreground">
            Current active alerts
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">System Health</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{systemHealth}</div>
          <p className="text-xs text-muted-foreground">
            Overall system health status
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

