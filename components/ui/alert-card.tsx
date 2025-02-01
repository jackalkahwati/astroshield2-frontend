import { AlertCircle, AlertTriangle, CheckCircle2, Info } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface AlertCardProps {
  title: string
  description: string
  variant?: "default" | "destructive" | "warning" | "success"
  className?: string
}

export function AlertCard({ title, description, variant = "default", className }: AlertCardProps) {
  const icons = {
    default: Info,
    destructive: AlertCircle,
    warning: AlertTriangle,
    success: CheckCircle2,
  }

  const Icon = icons[variant]

  return (
    <Card
      className={cn(
        "transition-all hover:shadow-lg",
        {
          "border-destructive": variant === "destructive",
          "border-warning": variant === "warning",
          "border-success": variant === "success",
        },
        className,
      )}
    >
      <CardHeader className="flex flex-row items-center gap-2 space-y-0 pb-2">
        <Icon
          className={cn("h-4 w-4", {
            "text-destructive": variant === "destructive",
            "text-warning": variant === "warning",
            "text-success": variant === "success",
          })}
        />
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <CardDescription>{description}</CardDescription>
      </CardContent>
    </Card>
  )
}

