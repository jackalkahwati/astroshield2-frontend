"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { useToast } from "@/components/ui/use-toast"

export function SettingsForm() {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [displayUnits, setDisplayUnits] = useState("metric")
  const [language, setLanguage] = useState("english")
  const [updateInterval, setUpdateInterval] = useState("5")
  const [dataRetention, setDataRetention] = useState("30")
  const [enableNotifications, setEnableNotifications] = useState(true)
  const [timeZone, setTimeZone] = useState("utc")
  const [defaultChartType, setDefaultChartType] = useState("line")
  const [collisionThreshold, setCollisionThreshold] = useState("5")
  const [anomalySensitivity, setAnomalySensitivity] = useState("medium")
  const [alertCategories, setAlertCategories] = useState({
    collisions: true,
    maneuvers: true,
    system: true,
  })
  const [auditLogRetention, setAuditLogRetention] = useState("90")
  const [exportFormat, setExportFormat] = useState("csv")
  const { toast } = useToast()

  const handleSaveSettings = () => {
    // Here you would typically save the settings to your backend
    toast({
      title: "Settings saved",
      description: "Your settings have been saved successfully.",
    })
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Display Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Display Units</Label>
              <Select value={displayUnits} onValueChange={setDisplayUnits}>
                <SelectTrigger>
                  <SelectValue placeholder="Select display units" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="metric">Metric</SelectItem>
                  <SelectItem value="imperial">Imperial</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Time Zone</Label>
              <Select value={timeZone} onValueChange={setTimeZone}>
                <SelectTrigger>
                  <SelectValue placeholder="Select time zone" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="utc">UTC</SelectItem>
                  <SelectItem value="local">Local Time</SelectItem>
                  <SelectItem value="custom">Custom...</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Language</Label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger>
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="english">English</SelectItem>
                  <SelectItem value="spanish">Spanish</SelectItem>
                  <SelectItem value="french">French</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Default Chart Type</Label>
              <Select value={defaultChartType} onValueChange={setDefaultChartType}>
                <SelectTrigger>
                  <SelectValue placeholder="Select chart type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="line">Line Chart</SelectItem>
                  <SelectItem value="scatter">Scatter Plot</SelectItem>
                  <SelectItem value="bar">Bar Chart</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data Settings</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="update-interval">Update Interval (seconds)</Label>
            <Input
              id="update-interval"
              type="number"
              value={updateInterval}
              onChange={(e) => setUpdateInterval(e.target.value)}
              min="1"
              max="60"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="data-retention">Data Retention (days)</Label>
            <Input
              id="data-retention"
              type="number"
              value={dataRetention}
              onChange={(e) => setDataRetention(e.target.value)}
              min="1"
              max="365"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Alert Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Collision Risk Threshold (%)</Label>
              <Input
                type="number"
                value={collisionThreshold}
                onChange={(e) => setCollisionThreshold(e.target.value)}
                min="0"
                max="100"
              />
            </div>
            <div className="space-y-2">
              <Label>Anomaly Detection Sensitivity</Label>
              <Select value={anomalySensitivity} onValueChange={setAnomalySensitivity}>
                <SelectTrigger>
                  <SelectValue placeholder="Select sensitivity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="space-y-2">
            <Label>Alert Categories</Label>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex items-center space-x-2">
                <Switch
                  id="collision-alerts"
                  checked={alertCategories.collisions}
                  onCheckedChange={(checked) => setAlertCategories((prev) => ({ ...prev, collisions: checked }))}
                />
                <Label htmlFor="collision-alerts">Collision Risks</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="maneuver-alerts"
                  checked={alertCategories.maneuvers}
                  onCheckedChange={(checked) => setAlertCategories((prev) => ({ ...prev, maneuvers: checked }))}
                />
                <Label htmlFor="maneuver-alerts">Maneuvers</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="system-alerts"
                  checked={alertCategories.system}
                  onCheckedChange={(checked) => setAlertCategories((prev) => ({ ...prev, system: checked }))}
                />
                <Label htmlFor="system-alerts">System Status</Label>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>System Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input type="password" value="************************" readOnly />
              <Button variant="outline" size="sm">
                Regenerate Key
              </Button>
            </div>
            <div className="space-y-2">
              <Label>Audit Log Retention (days)</Label>
              <Input
                type="number"
                value={auditLogRetention}
                onChange={(e) => setAuditLogRetention(e.target.value)}
                min="1"
                max="365"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label>Default Export Format</Label>
            <Select value={exportFormat} onValueChange={setExportFormat}>
              <SelectTrigger>
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="json">JSON</SelectItem>
                <SelectItem value="xlsx">Excel</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button size="lg" className="bg-blue-500 hover:bg-blue-600" onClick={handleSaveSettings}>
          SAVE SETTINGS
        </Button>
      </div>
    </div>
  )
}

