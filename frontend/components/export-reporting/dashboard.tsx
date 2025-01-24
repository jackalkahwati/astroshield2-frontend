"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatePickerWithRange } from "@/components/ui/date-range-picker"
import { Loader2 } from "lucide-react"
import { toast } from "@/components/ui/use-toast"

export function ExportReportingDashboard() {
  const [exportFormat, setExportFormat] = useState<string>("pdf")
  const [dateRange, setDateRange] = useState<{ from: Date; to: Date } | undefined>()
  const [isExporting, setIsExporting] = useState(false)

  const handleExport = async () => {
    if (!dateRange) {
      toast({
        title: "Date range required",
        description: "Please select a date range for your export.",
        variant: "destructive",
      })
      return
    }

    setIsExporting(true)
    // Simulating export process
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setIsExporting(false)

    toast({
      title: "Export Successful",
      description: `Your ${exportFormat.toUpperCase()} report has been generated.`,
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Generate Report</CardTitle>
        <CardDescription>Export your data in various formats</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Export Format</label>
            <Select value={exportFormat} onValueChange={setExportFormat}>
              <SelectTrigger>
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pdf">PDF</SelectItem>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="xlsx">Excel</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Date Range</label>
            <DatePickerWithRange date={dateRange} setDate={setDateRange} />
          </div>
        </div>
        <Button onClick={handleExport} disabled={isExporting}>
          {isExporting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Exporting...
            </>
          ) : (
            "Generate Report"
          )}
        </Button>
      </CardContent>
    </Card>
  )
}

