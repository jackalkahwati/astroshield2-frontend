"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Settings, User, Shield, Activity } from "lucide-react"

export default function SettingsPage() {
  const router = useRouter()
  const [profile, setProfile] = useState({
    name: "John Doe",
    email: "john@example.com",
    role: "Administrator"
  })

  // Simple metrics following the established pattern
  const metrics = [
    { title: "API Status", value: "Active", icon: Activity },
    { title: "Profile Status", value: "Complete", icon: User },
    { title: "Security Level", value: "High", icon: Shield },
    { title: "Theme Mode", value: "Dark", icon: Settings },
  ]

  // Settings configuration data
  const configurationData = [
    {
      setting: "User Profile",
      value: `${profile.name} (${profile.email})`,
      status: "configured",
      type: "Profile",
      lastModified: "Jun 6, 2025, 6:41 PM"
    },
    {
      setting: "Theme Preference",
      value: "Dark Mode",
      status: "active",
      type: "Display",
      lastModified: "Jun 5, 2025, 2:30 PM"
    },
    {
      setting: "API Configuration",
      value: "Endpoints Configured",
      status: "active",
      type: "Integration",
      lastModified: "Jun 4, 2025, 10:15 AM"
    },
    {
      setting: "Security Settings",
      value: "Two-Factor Enabled",
      status: "secured",
      type: "Security",
      lastModified: "Jun 3, 2025, 4:20 PM"
    },
    {
      setting: "Notification Preferences",
      value: "Email + SMS",
      status: "configured",
      type: "Communication",
      lastModified: "Jun 2, 2025, 11:45 AM"
    },
  ]

  // System preferences data
  const systemPreferences = [
    {
      preference: "Auto-Save Frequency",
      value: "Every 5 minutes",
      category: "Performance",
      status: "default"
    },
    {
      preference: "Data Retention",
      value: "90 days",
      category: "Storage",
      status: "custom"
    },
    {
      preference: "Session Timeout",
      value: "30 minutes",
      category: "Security",
      status: "default"
    },
    {
      preference: "Export Format",
      value: "JSON + CSV",
      category: "Data",
      status: "custom"
    },
    {
      preference: "Language",
      value: "English (US)",
      category: "Localization",
      status: "default"
    },
  ]

  const getStatusVariant = (status: string) => {
    switch (status) {
      case "active": case "configured": case "secured": return "default"
      case "warning": return "destructive"
      case "disabled": return "secondary"
      case "custom": return "outline"
      default: return "outline"
    }
  }

  const handleLogout = () => {
    // Clear any stored tokens/session data
    if (typeof window !== 'undefined') {
      localStorage.clear()
      sessionStorage.clear()
    }
    router.push("/login")
  }

  const handleExportSettings = () => {
    const settingsData = {
      profile,
      configurationData,
      systemPreferences,
      exportedAt: new Date().toISOString(),
      version: "1.0"
    }
    
    const dataStr = JSON.stringify(settingsData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `astroshield-settings-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const handleResetPreferences = () => {
    if (window.confirm('Are you sure you want to reset all preferences to default values? This action cannot be undone.')) {
      // Reset profile to defaults
      setProfile({
        name: "John Doe",
        email: "john@example.com",
        role: "Administrator"
      })
      
      // In a real app, you would also reset other preferences
      alert('Preferences have been reset to default values.')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Settings</h1>
      </div>

      {/* Key Metrics Cards - same pattern as other pages */}
      <div className="grid gap-4 md:grid-cols-4">
        {metrics.map((metric) => {
          const Icon = metric.icon
          return (
            <Card key={metric.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-white">{metric.title}</CardTitle>
                <Icon className="h-4 w-4 text-white" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{metric.value}</div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Configuration Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Configuration Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Setting</TableHead>
                <TableHead>Current Value</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Last Modified</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {configurationData.map((config, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{config.setting}</TableCell>
                  <TableCell>{config.value}</TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(config.status)}>
                      {config.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{config.type}</Badge>
                  </TableCell>
                  <TableCell className="text-sm">{config.lastModified}</TableCell>
                  <TableCell>
                    <Button variant="outline" size="sm">
                      Edit
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* System Preferences */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">System Preferences</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Preference</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {systemPreferences.map((pref, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{pref.preference}</TableCell>
                  <TableCell>{pref.value}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{pref.category}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={getStatusVariant(pref.status)}>
                      {pref.status.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Button variant="outline" size="sm">
                      Modify
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-white">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Link href="/settings/api">
              <Button>
                API Documentation
              </Button>
            </Link>
            <Button variant="outline" onClick={handleExportSettings}>
              Export Settings
            </Button>
            <Button variant="outline" onClick={handleResetPreferences}>
              Reset Preferences
            </Button>
            <Button 
              variant="destructive"
              onClick={handleLogout}
            >
              Logout
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

