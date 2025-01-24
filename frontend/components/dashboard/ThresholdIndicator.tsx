import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ThresholdIndicatorProps {
  title: string
  threshold: string
  description: string
  passCriteria: string
  features: string[]
}

export function ThresholdIndicator({ title, threshold, description, passCriteria, features }: ThresholdIndicatorProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold">{title}</CardTitle>
          <Badge variant="secondary">Threshold: {threshold}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">{description}</p>
        <div className="mt-2">
          <p className="text-sm font-medium">Pass Criteria</p>
          <p className="text-sm">{passCriteria}</p>
        </div>
        <div className="mt-2">
          <p className="text-sm font-medium">Features</p>
          <ul className="list-disc list-inside text-sm">
            {features.map((feature, index) => (
              <li key={index}>{feature}</li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}

