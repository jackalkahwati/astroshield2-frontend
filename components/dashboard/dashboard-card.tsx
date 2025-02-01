import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ReactNode } from "react"
import { cn } from "@/lib/utils"

export interface DashboardCardProps {
  title: string
  children?: ReactNode
  className?: string
}

export function DashboardCard({ title, children, className }: DashboardCardProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  )
} 