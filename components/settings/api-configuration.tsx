'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Settings,
  Plus,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Download,
  Upload,
  Key,
  Globe,
  Database,
  FileJson,
  FileSpreadsheet,
  FileText
} from 'lucide-react'
import { apiManager, VendorConfig, DataExporter } from '@/lib/api-integration'

interface APIEndpoint {
  id: string
  name: string
  vendor: VendorConfig['vendor']
  baseUrl: string
  apiKey?: string
  enabled: boolean
  status: 'connected' | 'disconnected' | 'error'
  lastChecked?: Date
  dataTypes: string[]
}

export default function APIConfiguration() {
  const [endpoints, setEndpoints] = useState<APIEndpoint[]>([])
  const [newEndpoint, setNewEndpoint] = useState<Partial<APIEndpoint>>({
    vendor: 'sda',
    enabled: true,
    dataTypes: []
  })
  const [exportSettings, setExportSettings] = useState({
    autoExport: false,
    exportInterval: 3600, // seconds
    exportFormats: ['json', 'csv'],
    includeMetadata: true
  })
  const [isTestingConnection, setIsTestingConnection] = useState<string | null>(null)

  // Load saved endpoints
  useEffect(() => {
    const savedEndpoints = localStorage.getItem('astroshield_api_endpoints')
    if (savedEndpoints) {
      setEndpoints(JSON.parse(savedEndpoints))
    } else {
      // Default endpoints
      setEndpoints([
        {
          id: 'sda-primary',
          name: 'SDA Primary',
          vendor: 'sda',
          baseUrl: process.env.NEXT_PUBLIC_SDA_API_URL || 'https://api.sda.mil',
          enabled: true,
          status: 'disconnected',
          dataTypes: ['objects', 'conjunctions', 'threats']
        },
        {
          id: 'leolabs-1',
          name: 'LeoLabs API',
          vendor: 'leolabs',
          baseUrl: 'https://api.leolabs.space/v1',
          enabled: false,
          status: 'disconnected',
          dataTypes: ['conjunctions', 'states']
        }
      ])
    }
  }, [])

  // Save endpoints when changed
  useEffect(() => {
    localStorage.setItem('astroshield_api_endpoints', JSON.stringify(endpoints))
    
    // Register with API manager
    endpoints.forEach(endpoint => {
      if (endpoint.enabled) {
        apiManager.registerClient(endpoint.id, {
          vendor: endpoint.vendor,
          baseUrl: endpoint.baseUrl,
          apiKey: endpoint.apiKey,
          dataFormat: 'json',
          authentication: endpoint.apiKey ? 'api_key' : 'none'
        })
      }
    })
  }, [endpoints])

  const testConnection = async (endpointId: string) => {
    setIsTestingConnection(endpointId)
    const endpoint = endpoints.find(e => e.id === endpointId)
    if (!endpoint) return

    try {
      // Simulate connection test
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // In real implementation, would test actual endpoint
      const success = Math.random() > 0.3
      
      setEndpoints(prev => prev.map(e => 
        e.id === endpointId 
          ? { 
              ...e, 
              status: success ? 'connected' : 'error',
              lastChecked: new Date()
            }
          : e
      ))
    } catch (error) {
      setEndpoints(prev => prev.map(e => 
        e.id === endpointId 
          ? { ...e, status: 'error', lastChecked: new Date() }
          : e
      ))
    } finally {
      setIsTestingConnection(null)
    }
  }

  const addEndpoint = () => {
    if (!newEndpoint.name || !newEndpoint.baseUrl) return

    const endpoint: APIEndpoint = {
      id: `${newEndpoint.vendor}-${Date.now()}`,
      name: newEndpoint.name,
      vendor: newEndpoint.vendor || 'custom',
      baseUrl: newEndpoint.baseUrl,
      apiKey: newEndpoint.apiKey,
      enabled: true,
      status: 'disconnected',
      dataTypes: newEndpoint.dataTypes || []
    }

    setEndpoints(prev => [...prev, endpoint])
    setNewEndpoint({ vendor: 'sda', enabled: true, dataTypes: [] })
  }

  const removeEndpoint = (id: string) => {
    setEndpoints(prev => prev.filter(e => e.id !== id))
  }

  const toggleEndpoint = (id: string) => {
    setEndpoints(prev => prev.map(e => 
      e.id === id ? { ...e, enabled: !e.enabled } : e
    ))
  }

  const exportConfiguration = () => {
    const config = {
      endpoints,
      exportSettings,
      exportedAt: new Date().toISOString()
    }
    DataExporter.exportToJSON(config, 'astroshield-api-config.json')
  }

  const importConfiguration = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target?.result as string)
        if (config.endpoints) {
          setEndpoints(config.endpoints)
        }
        if (config.exportSettings) {
          setExportSettings(config.exportSettings)
        }
      } catch (error) {
        console.error('Failed to import configuration:', error)
      }
    }
    reader.readAsText(file)
  }

  const getStatusIcon = (status: APIEndpoint['status']) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      default:
        return <div className="h-4 w-4 rounded-full bg-gray-300" />
    }
  }

  const getVendorIcon = (vendor: VendorConfig['vendor']) => {
    switch (vendor) {
      case 'sda':
        return <Database className="h-4 w-4" />
      case 'leolabs':
      case 'analytical_graphics':
      case 'raytheon':
        return <Globe className="h-4 w-4" />
      default:
        return <Settings className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">API Configuration</h2>
          <p className="text-sm text-gray-600 mt-1">
            Manage multi-vendor integrations and export settings
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={exportConfiguration}>
            <Download className="h-4 w-4 mr-1" />
            Export Config
          </Button>
          <Label htmlFor="import-config" className="cursor-pointer">
            <Button variant="outline" size="sm" asChild>
              <span>
                <Upload className="h-4 w-4 mr-1" />
                Import Config
              </span>
            </Button>
          </Label>
          <Input
            id="import-config"
            type="file"
            accept=".json"
            className="hidden"
            onChange={importConfiguration}
          />
        </div>
      </div>

      <Tabs defaultValue="endpoints">
        <TabsList>
          <TabsTrigger value="endpoints">API Endpoints</TabsTrigger>
          <TabsTrigger value="export">Export Settings</TabsTrigger>
          <TabsTrigger value="status">System Status</TabsTrigger>
        </TabsList>

        <TabsContent value="endpoints" className="space-y-4">
          {/* Configured Endpoints */}
          <Card>
            <CardHeader>
              <CardTitle>Configured Endpoints</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {endpoints.map(endpoint => (
                  <div key={endpoint.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          {getVendorIcon(endpoint.vendor)}
                          <h4 className="font-medium">{endpoint.name}</h4>
                          {getStatusIcon(endpoint.status)}
                          <Badge variant={endpoint.enabled ? "default" : "outline"}>
                            {endpoint.enabled ? 'Enabled' : 'Disabled'}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600">{endpoint.baseUrl}</p>
                        <div className="flex items-center gap-2 text-sm">
                          <Key className="h-3 w-3" />
                          <span>{endpoint.apiKey ? 'API Key configured' : 'No authentication'}</span>
                        </div>
                        {endpoint.dataTypes.length > 0 && (
                          <div className="flex gap-2">
                            {endpoint.dataTypes.map(type => (
                              <Badge key={type} variant="outline" className="text-xs">
                                {type}
                              </Badge>
                            ))}
                          </div>
                        )}
                        {endpoint.lastChecked && (
                          <p className="text-xs text-gray-500">
                            Last checked: {endpoint.lastChecked.toLocaleString()}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={endpoint.enabled}
                          onCheckedChange={() => toggleEndpoint(endpoint.id)}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => testConnection(endpoint.id)}
                          disabled={isTestingConnection === endpoint.id}
                        >
                          {isTestingConnection === endpoint.id ? 'Testing...' : 'Test'}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => removeEndpoint(endpoint.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Add New Endpoint */}
          <Card>
            <CardHeader>
              <CardTitle>Add New Endpoint</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={newEndpoint.name || ''}
                    onChange={e => setNewEndpoint(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="My API Endpoint"
                  />
                </div>
                <div>
                  <Label htmlFor="vendor">Vendor</Label>
                  <select
                    id="vendor"
                    className="w-full rounded-md border border-gray-300 px-3 py-2"
                    value={newEndpoint.vendor}
                    onChange={e => setNewEndpoint(prev => ({ 
                      ...prev, 
                      vendor: e.target.value as VendorConfig['vendor'] 
                    }))}
                  >
                    <option value="sda">SDA</option>
                    <option value="leolabs">LeoLabs</option>
                    <option value="analytical_graphics">Analytical Graphics</option>
                    <option value="raytheon">Raytheon</option>
                    <option value="custom">Custom</option>
                  </select>
                </div>
                <div>
                  <Label htmlFor="baseUrl">Base URL</Label>
                  <Input
                    id="baseUrl"
                    value={newEndpoint.baseUrl || ''}
                    onChange={e => setNewEndpoint(prev => ({ ...prev, baseUrl: e.target.value }))}
                    placeholder="https://api.example.com"
                  />
                </div>
                <div>
                  <Label htmlFor="apiKey">API Key (Optional)</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    value={newEndpoint.apiKey || ''}
                    onChange={e => setNewEndpoint(prev => ({ ...prev, apiKey: e.target.value }))}
                    placeholder="Your API key"
                  />
                </div>
              </div>
              <Button className="mt-4" onClick={addEndpoint}>
                <Plus className="h-4 w-4 mr-1" />
                Add Endpoint
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="export" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Export Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auto-export">Automatic Export</Label>
                  <p className="text-sm text-gray-600">
                    Automatically export data at regular intervals
                  </p>
                </div>
                <Switch
                  id="auto-export"
                  checked={exportSettings.autoExport}
                  onCheckedChange={checked => 
                    setExportSettings(prev => ({ ...prev, autoExport: checked }))
                  }
                />
              </div>

              {exportSettings.autoExport && (
                <div>
                  <Label htmlFor="export-interval">Export Interval (seconds)</Label>
                  <Input
                    id="export-interval"
                    type="number"
                    value={exportSettings.exportInterval}
                    onChange={e => 
                      setExportSettings(prev => ({ 
                        ...prev, 
                        exportInterval: parseInt(e.target.value) || 3600 
                      }))
                    }
                  />
                </div>
              )}

              <div>
                <Label>Export Formats</Label>
                <div className="flex gap-4 mt-2">
                  {['json', 'csv', 'stk'].map(format => (
                    <label key={format} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={exportSettings.exportFormats.includes(format)}
                        onChange={e => {
                          if (e.target.checked) {
                            setExportSettings(prev => ({
                              ...prev,
                              exportFormats: [...prev.exportFormats, format]
                            }))
                          } else {
                            setExportSettings(prev => ({
                              ...prev,
                              exportFormats: prev.exportFormats.filter(f => f !== format)
                            }))
                          }
                        }}
                      />
                      <span className="flex items-center gap-1">
                        {format === 'json' && <FileJson className="h-4 w-4" />}
                        {format === 'csv' && <FileSpreadsheet className="h-4 w-4" />}
                        {format === 'stk' && <FileText className="h-4 w-4" />}
                        {format.toUpperCase()}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="include-metadata">Include Metadata</Label>
                  <p className="text-sm text-gray-600">
                    Add timestamps, sources, and validation info to exports
                  </p>
                </div>
                <Switch
                  id="include-metadata"
                  checked={exportSettings.includeMetadata}
                  onCheckedChange={checked => 
                    setExportSettings(prev => ({ ...prev, includeMetadata: checked }))
                  }
                />
              </div>
            </CardContent>
          </Card>

          <Alert>
            <Settings className="h-4 w-4" />
            <AlertTitle>Export Configuration</AlertTitle>
            <AlertDescription>
              Exports will be saved to your default downloads folder. For automated exports,
              ensure your browser allows automatic downloads from this site.
            </AlertDescription>
          </Alert>
        </TabsContent>

        <TabsContent value="status" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Integration Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold">
                      {endpoints.filter(e => e.status === 'connected').length}
                    </p>
                    <p className="text-sm text-gray-600">Connected APIs</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">
                      {endpoints.filter(e => e.enabled).length}
                    </p>
                    <p className="text-sm text-gray-600">Enabled Endpoints</p>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-medium mb-2">Data Sources</h4>
                  <div className="space-y-2">
                    {['objects', 'conjunctions', 'threats', 'states', 'maneuvers'].map(dataType => {
                      const sources = endpoints.filter(e => 
                        e.enabled && e.dataTypes.includes(dataType)
                      )
                      return (
                        <div key={dataType} className="flex items-center justify-between">
                          <span className="text-sm capitalize">{dataType}</span>
                          <Badge variant={sources.length > 0 ? "default" : "outline"}>
                            {sources.length} source{sources.length !== 1 ? 's' : ''}
                          </Badge>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
} 