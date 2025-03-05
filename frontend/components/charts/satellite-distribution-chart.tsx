"use client"

import { useState } from "react"
import { Bar, Pie, Doughnut, PolarArea } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Register all the chart components
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  ArcElement,
  RadialLinearScale,
  Title, 
  Tooltip, 
  Legend
)

type ChartType = "bar" | "pie" | "doughnut" | "polar"

export function SatelliteDistributionChart() {
  const [chartType, setChartType] = useState<ChartType>("bar")
  
  // Satellite distribution data
  const labels = ["LEO", "MEO", "GEO", "HEO", "Other"]
  const values = [8, 2, 1, 1, 0]
  
  // Common data for all chart types
  const chartData = {
    labels,
    datasets: [
      {
        label: "Satellites",
        data: values,
        backgroundColor: [
          "rgba(54, 162, 235, 0.6)",
          "rgba(75, 192, 192, 0.6)",
          "rgba(255, 206, 86, 0.6)",
          "rgba(153, 102, 255, 0.6)",
          "rgba(255, 99, 132, 0.6)",
        ],
        borderColor: [
          "rgba(54, 162, 235, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(153, 102, 255, 1)",
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
        text: "Satellite Distribution by Orbit",
      },
    },
  }
  
  // Render the appropriate chart based on the selected type
  const renderChart = () => {
    switch (chartType) {
      case "bar":
        return <Bar data={chartData} options={options} />
      case "pie":
        return <Pie data={chartData} options={options} />
      case "doughnut":
        return <Doughnut data={chartData} options={options} />
      case "polar":
        return <PolarArea data={chartData} options={options} />
      default:
        return <Bar data={chartData} options={options} />
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
            <SelectItem value="bar">Bar Chart</SelectItem>
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