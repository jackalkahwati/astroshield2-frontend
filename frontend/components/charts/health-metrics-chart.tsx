"use client"

import { useState } from "react"
import { Line, Pie, Doughnut, PolarArea } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Register all the chart components
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  ArcElement,
  RadialLinearScale,
  Title, 
  Tooltip, 
  Legend,
  Filler
)

type ChartType = "line" | "pie" | "doughnut" | "polar"

export function HealthMetricsChart() {
  const [chartType, setChartType] = useState<ChartType>("doughnut")
  
  // Health metrics data
  const labels = ["System Uptime", "Response Time", "Error Rate", "CPU Usage", "Memory Usage"]
  const values = [98, 95, 97, 85, 90]
  
  // Line chart data
  const lineData = {
    labels,
    datasets: [
      {
        label: "Current (%)",
        data: values,
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Previous (%)",
        data: [96, 92, 95, 88, 93],
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
        fill: true,
        tension: 0.4,
      }
    ],
  }
  
  // Pie, Doughnut, and Polar Area chart data
  const circularData = {
    labels,
    datasets: [
      {
        label: "Health Metrics (%)",
        data: values,
        backgroundColor: [
          "rgba(75, 192, 192, 0.6)",
          "rgba(54, 162, 235, 0.6)",
          "rgba(153, 102, 255, 0.6)",
          "rgba(255, 206, 86, 0.6)",
          "rgba(255, 99, 132, 0.6)",
        ],
        borderColor: [
          "rgba(75, 192, 192, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(153, 102, 255, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(255, 99, 132, 1)",
        ],
        borderWidth: 1,
      },
    ],
  }
  
  // Common options for all chart types
  const options = { 
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "System Health Metrics",
      },
    },
  }
  
  // Render the appropriate chart based on the selected type
  const renderChart = () => {
    switch (chartType) {
      case "line":
        return <Line data={lineData} options={options} />
      case "pie":
        return <Pie data={circularData} options={options} />
      case "doughnut":
        return <Doughnut data={circularData} options={options} />
      case "polar":
        return <PolarArea data={circularData} options={options} />
      default:
        return <Doughnut data={circularData} options={options} />
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Select value={chartType} onValueChange={(value) => setChartType(value as ChartType)}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select chart type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="line">Line Chart</SelectItem>
            <SelectItem value="pie">Pie Chart</SelectItem>
            <SelectItem value="doughnut">Doughnut Chart</SelectItem>
            <SelectItem value="polar">Polar Area</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="h-[250px]">
        {renderChart()}
      </div>
    </div>
  )
} 