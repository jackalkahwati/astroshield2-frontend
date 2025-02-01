import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface MLIndicatorProps {
  title: string
  version: string
  description: string
  confidenceScore: string
  trainingAccuracy: string
  algorithm: string
  passCriteria: string
  features: string[]
}

export function MLIndicator({
  title,
  version,
  description,
  confidenceScore,
  trainingAccuracy,
  algorithm,
  passCriteria,
  features,
}: MLIndicatorProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold">{title}</CardTitle>
          <Badge variant="secondary">{version}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">{description}</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium">Confidence Score</p>
            <p className="text-2xl font-bold">{confidenceScore}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Training Accuracy</p>
            <p className="text-2xl font-bold">{trainingAccuracy}</p>
          </div>
        </div>
        <div className="mt-4">
          <p className="text-sm font-medium">Algorithm</p>
          <p className="text-sm">{algorithm}</p>
        </div>
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

