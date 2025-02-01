import { ResponsiveContainer, LineChart as RechartsLineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

interface LineChartProps {
  data: any[]
  width?: number | string
  height?: number | string
  dataKey: string
  xAxisKey?: string
  stroke?: string
  className?: string
}

export function LineChart({
  data,
  width = "100%",
  height = 300,
  dataKey,
  xAxisKey = "timestamp",
  stroke = "#2563eb",
  className
}: LineChartProps) {
  return (
    <ResponsiveContainer width={width} height={height} className={className}>
      <RechartsLineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <Line type="monotone" dataKey={dataKey} stroke={stroke} strokeWidth={2} dot={false} />
        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" opacity={0.2} />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
      </RechartsLineChart>
    </ResponsiveContainer>
  )
} 