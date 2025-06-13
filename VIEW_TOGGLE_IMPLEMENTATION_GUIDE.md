# View Toggle Implementation Guide

## Overview
This guide shows how to implement list/graph view toggles consistently across all AstroShield pages with lists and tables.

## Core Pattern

### 1. Import Required Components
```tsx
import { ViewToggle, useViewToggle } from "@/components/ui/view-toggle"
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts'
```

### 2. Setup View Toggle Hook
```tsx
export default function YourPage() {
  const { viewMode, setViewMode, isListView, isGraphView } = useViewToggle("graph") // Default to graph
  
  // ... your existing data
}
```

### 3. Add Toggle to Card Header
```tsx
<Card>
  <CardHeader className="flex flex-row items-center justify-between">
    <CardTitle className="text-white">Your Data Table</CardTitle>
    <ViewToggle currentView={viewMode} onViewChange={setViewMode} />
  </CardHeader>
  <CardContent>
    {isGraphView ? (
      // Graph view content
    ) : (
      // List view content (existing table)
    )}
  </CardContent>
</Card>
```

## Default Behavior

- **Dashboard**: Defaults to graph view ✅
- **All other pages**: Can default to graph or list based on use case
- **Consistent toggle UI**: Top-right of each card header

## Chart Types by Data Type

### System Status / Health Metrics
- **Bar Chart**: Health percentages, counts
- **Pie Chart**: Status distribution (operational, warning, critical)

### Time Series Data
- **Line Chart**: Events over time, confidence trends
- **Area Chart**: With uncertainty bands

### Categorical Data
- **Bar Chart**: Counts by category, performance metrics
- **Scatter Plot**: Correlation analysis (uncertainty vs count)

### Recent Events/Activities
- **Line Chart**: Timeline view with metrics
- **Bar Chart**: Event counts, severity levels

## Implementation Status

### ✅ Completed
- **Dashboard** (`/dashboard`) - System status & alerts with graphs
- **Analytics** (`/analytics`) - Classification accuracy & event timeline
- **Maneuvers** (`/maneuvers`) - Maneuver summaries & status distribution
- **Trajectory Analysis** (`/trajectory-analysis`) - Already has advanced charts

### 🔄 Next to Implement
- **Event Correlation** (`/event-correlation`) - Events list → timeline charts
- **Proximity Operations** (`/proximity-operations`) - Proximity events → scatter plots
- **CCDM Analysis** (`/ccdm`) - Conjunction events → risk timeline
- **Kafka Monitor** (`/kafka-monitor`) - Message lists → throughput charts
- **Protection** (`/protection`) - Asset lists → threat distribution
- **Decision Support** (`/decision-support`) - Workflows → progress charts

## Data Preparation Pattern

```tsx
// Prepare data for charts
const chartData = rawData.map(item => ({
  name: item.name.split(' ')[0], // Shortened names
  value: parseFloat(item.value),
  fill: item.status === 'good' ? '#10B981' : '#EF4444' // Color coding
}))
```

## Chart Configuration

### Standard Colors
- **Success/Good**: `#10B981` (green)
- **Warning/Medium**: `#F59E0B` (yellow)  
- **Error/High**: `#EF4444` (red)
- **Info/Neutral**: `#6B7280` (gray)
- **Primary**: `#8B5CF6` (purple)

### Consistent Styling
```tsx
<Tooltip 
  contentStyle={{
    backgroundColor: '#1F2937',
    border: '1px solid #374151',
    borderRadius: '6px',
    color: '#F9FAFB'
  }}
/>
```

## Usage Examples

### Simple Bar Chart
```tsx
<ResponsiveContainer width="100%" height="100%">
  <BarChart data={chartData}>
    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
    <XAxis dataKey="name" stroke="#9CA3AF" fontSize={12} />
    <YAxis stroke="#9CA3AF" fontSize={12} />
    <Tooltip />
    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
  </BarChart>
</ResponsiveContainer>
```

### Time Series Line Chart
```tsx
<LineChart data={timeSeriesData}>
  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
  <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
  <YAxis stroke="#9CA3AF" fontSize={12} />
  <Tooltip />
  <Line 
    type="monotone" 
    dataKey="value" 
    stroke="#10B981" 
    strokeWidth={3}
    dot={{ fill: '#10B981', strokeWidth: 2, r: 6 }}
  />
</LineChart>
```

This pattern ensures consistent UX across all AstroShield operational interfaces while providing space operators with both detailed list views and visual graph insights. 