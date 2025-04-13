"use client"

import { useState } from "react"
import { Bar, Line, Pie, Radar, Doughnut } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
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
  BarElement, 
  PointElement, 
  LineElement, 
  ArcElement,
  RadialLinearScale,
  Title, 
  Tooltip, 
  Legend,
  Filler
)

type ChartType = "bar" | "line" | "pie" | "radar" | "doughnut"

export function ActivityChart() {
  const [chartType, setChartType] = useState<ChartType>("bar")
  
  // Common data for all chart types
  const labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
  const values = [1, 4, 2, 5, 3]
  
  // Bar and Line chart data
  const timeSeriesData = {
    labels,
    datasets: [
      {
        label: "Alerts",
        data: values,
        backgroundColor: "rgba(255,99,132,0.6)",
        borderColor: "rgba(255,99,132,1)",
        borderWidth: 1,
        fill: chartType === "line" ? "origin" : undefined,
      },
    ],
  }
  
  // Pie and Doughnut chart data
  const pieData = {
    labels,
    datasets: [
      {
        label: "Alerts",
        data: values,
        backgroundColor: [
          "rgba(255, 99, 132, 0.6)",
          "rgba(54, 162, 235, 0.6)",
          "rgba(255, 206, 86, 0.6)",
          "rgba(75, 192, 192, 0.6)",
          "rgba(153, 102, 255, 0.6)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(153, 102, 255, 1)",
        ],
        borderWidth: 1,
      },
    ],
  }
  
  // Radar chart data
  const radarData = {
    labels,
    datasets: [
      {
        label: "Alerts",
        data: values,
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        borderColor: "rgba(255, 99, 132, 1)",
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
        text: "System Activity",
      },
    },
  }
  
  // Render the appropriate chart based on the selected type
  const renderChart = () => {
    switch (chartType) {
      case "bar":
        return <Bar data={timeSeriesData} options={options} />
      case "line":
        return <Line data={timeSeriesData} options={options} />
      case "pie":
        return <Pie data={pieData} options={options} />
      case "radar":
        return <Radar data={radarData} options={options} />
      case "doughnut":
        return <Doughnut data={pieData} options={options} />
      default:
        return <Bar data={timeSeriesData} options={options} />
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
            <SelectItem value="line">Line Chart</SelectItem>
            <SelectItem value="pie">Pie Chart</SelectItem>
            <SelectItem value="radar">Radar Chart</SelectItem>
            <SelectItem value="doughnut">Doughnut Chart</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="h-[250px]">
        {renderChart()}
      </div>
    </div>
  )
} 