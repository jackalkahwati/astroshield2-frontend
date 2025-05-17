import { Metadata } from 'next'
import DiagnosticsDashboard from '@/components/analytics/DiagnosticsDashboard'

export const metadata: Metadata = {
  title: 'Model Diagnostics',
  description: 'Quick-look charts that compare AstroShield predictions against TIP baseline',
}

export default function DiagnosticsPage() {
  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6">Model Diagnostics</h1>
      <DiagnosticsDashboard />
    </div>
  )
} 